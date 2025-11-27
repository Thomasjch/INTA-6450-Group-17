"""
Bayesian Email Scoring System
Integrates BM25 text scores with simplified Bayes calculation
validation and diagnostics too
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from math import gamma
import matplotlib.pyplot as plt

# ===== Configuration =====

DEFAULT_GRAPH_HOPS = 2
DEFAULT_SAME_COMMUNITY = 1
DEFAULT_ANOMALY_SCORE = 1.0

# ===== Data Classes =====
@dataclass
class Prior:
    """Prior probability that an email is related to wrongdoing"""
    pi: float = 1e-3

@dataclass
class ChannelParams:
    """Parameters for likelihood calculations"""
    text_alpha_pos: float = 5.0
    text_beta_pos: float = 2.0
    text_alpha_neg: float = 2.0
    text_beta_neg: float = 5.0
    time_scale_pos: float = 7 * 24 * 3600
    time_scale_neg: float = 30 * 24 * 3600
    graph_beta0: float = 0.0
    graph_beta: dict = None
    anom_mu_pos: float = 1.0
    anom_sigma_pos: float = 0.5
    anom_mu_neg: float = 0.0
    anom_sigma_neg: float = 0.5
    
    def __post_init__(self):
        if self.graph_beta is None:
            self.graph_beta = {
                "bias": 0.0,
                "neg_hops": 0.5,
                "same_comm": 0.8
            }

# ===== Likelihood Functions =====
def _beta_pdf(z, a, b):
    z = np.clip(z, 1e-6, 1-1e-6)
    B = gamma(a) * gamma(b) / gamma(a + b)
    return (z**(a-1)) * ((1-z)**(b-1)) / B

def _laplace_pdf(x, scale):
    return (1.0 / (2.0 * scale)) * np.exp(-np.abs(x) / scale)

def _lognormal_pdf(x, mu, sigma):
    x = np.maximum(x, 1e-12)
    return (1.0 / (x * sigma * np.sqrt(2 * np.pi))) * \
           np.exp(-((np.log(x) - mu)**2) / (2 * sigma**2))

def _sigmoid(u):
    return 1.0 / (1.0 + np.exp(-np.clip(u, -500, 500)))

def _safe_ratio(a, b, eps=1e-12):
    return (a + eps) / (b + eps)

# ===== Main Scoring Function =====
def score_emails_bayes(df: pd.DataFrame, 
                       prior: Prior, 
                       params: ChannelParams,
                       episode_ts: pd.Timestamp = None) -> pd.DataFrame:
    """
    Score emails using Bayesian approach with BM25 text scores
    """
    result = df.copy()

    if not np.issubdtype(result["email_ts"].dtype, np.datetime64):
        result["email_ts"] = pd.to_datetime(result["email_ts"], utc=True)

    if episode_ts is None:
        episode_ts = result['email_ts'].max()
        print(f"Using latest email timestamp as episode: {episode_ts}")
    
    result['time_delta_sec'] = (result['email_ts'] - episode_ts).dt.total_seconds()
    
    if 'anomaly_score' not in result.columns:
        result['anomaly_score'] = DEFAULT_ANOMALY_SCORE
        print(f"Warning: 'anomaly_score' not found, using default: {DEFAULT_ANOMALY_SCORE}")
    
    if 'graph_shortest_hops' not in result.columns:
        result['graph_shortest_hops'] = DEFAULT_GRAPH_HOPS
        print(f"Warning: 'graph_shortest_hops' not found, using default: {DEFAULT_GRAPH_HOPS}")
    
    if 'graph_same_community' not in result.columns:
        result['graph_same_community'] = DEFAULT_SAME_COMMUNITY
        print(f"Warning: 'graph_same_community' not found, using default: {DEFAULT_SAME_COMMUNITY}")
    
    result['text_score'] = result['text_score'].clip(0.0, 1.0)
    
    lr_text = _safe_ratio(
        _beta_pdf(result['text_score'], params.text_alpha_pos, params.text_beta_pos),
        _beta_pdf(result['text_score'], params.text_alpha_neg, params.text_beta_neg)
    )
    
    lr_time = _safe_ratio(
        _laplace_pdf(result['time_delta_sec'], params.time_scale_pos),
        _laplace_pdf(result['time_delta_sec'], params.time_scale_neg)
    )
    
    bias = np.ones(len(result))
    neg_hops = -result['graph_shortest_hops'].astype(float)
    same_comm = result['graph_same_community'].astype(float)
    
    lin = (params.graph_beta0 +
           params.graph_beta["bias"] * bias +
           params.graph_beta["neg_hops"] * neg_hops +
           params.graph_beta["same_comm"] * same_comm)
    
    s = np.clip(_sigmoid(lin), 1e-6, 1-1e-6)
    lr_graph = s / (1 - s)
    
    lr_anom = _safe_ratio(
        _lognormal_pdf(result['anomaly_score'], params.anom_mu_pos, params.anom_sigma_pos),
        _lognormal_pdf(result['anomaly_score'], params.anom_mu_neg, params.anom_sigma_neg)
    )
    
    prior_odds = prior.pi / (1 - prior.pi)
    posterior_odds = prior_odds * lr_text * lr_time * lr_graph * lr_anom
    posterior_prob = posterior_odds / (1 + posterior_odds)
    
    result['lr_text'] = lr_text
    result['lr_time'] = lr_time
    result['lr_graph'] = lr_graph
    result['lr_anom'] = lr_anom
    result['posterior_odds'] = posterior_odds
    result['posterior_prob'] = np.clip(posterior_prob, 0, 1)
    
    result = result.sort_values('posterior_prob', ascending=False).reset_index(drop=True)
    
    return result

# ===== Validation Functions =====
def validate_scoring(df: pd.DataFrame, show_plots: bool = True):
    """
    Validate Bayesian scoring with diagnostics and visualizations
    """
    
    print("="*80)
    print("BAYESIAN SCORING VALIDATION")
    print("="*80)
    
    print("\n1. POSTERIOR PROBABILITY STATISTICS")
    print("-" * 40)
    print(f"Mean posterior: {df['posterior_prob'].mean():.6f}")
    print(f"Median posterior: {df['posterior_prob'].median():.6f}")
    print(f"Std deviation: {df['posterior_prob'].std():.6f}")
    print(f"Min posterior: {df['posterior_prob'].min():.6f}")
    print(f"Max posterior: {df['posterior_prob'].max():.6f}")
    
    print("\n2. LIKELIHOOD RATIO ANALYSIS")
    print("-" * 40)
    for lr_col in ['lr_text', 'lr_time', 'lr_graph', 'lr_anom']:
        if lr_col in df.columns:
            print(f"\n{lr_col}:")
            print(f"  Mean: {df[lr_col].mean():.4f}")
            print(f"  Median: {df[lr_col].median():.4f}")
            print(f"  >1 (favors positive): {(df[lr_col] > 1).sum()} emails ({(df[lr_col] > 1).mean()*100:.1f}%)")
            print(f"  <1 (favors negative): {(df[lr_col] < 1).sum()} emails ({(df[lr_col] < 1).mean()*100:.1f}%)")
    
    print("\n3. FEATURE CORRELATION WITH POSTERIOR")
    print("-" * 40)
    features = ['text_score', 'lexicon_hits', 'phrase_score', 'temporal_score']
    for feat in features:
        if feat in df.columns:
            corr = df[[feat, 'posterior_prob']].corr().iloc[0, 1]
            print(f"{feat}: {corr:.4f}")
    
    print("\n4. TOP 10 EMAILS BY POSTERIOR PROBABILITY")
    print("-" * 40)
    top_10 = df.nlargest(10, 'posterior_prob')
    for idx, row in top_10.iterrows():
        print(f"\nRank {idx+1}: {row.get('email_id', 'N/A')}")
        print(f"  Posterior: {row['posterior_prob']:.6f}")
        print(f"  Text Score: {row.get('text_score', 0):.4f}")
        print(f"  Lexicon Hits: {row.get('lexicon_hits', 0)}")
        print(f"  Date: {row.get('email_ts', 'N/A')}")
    
    print("\n5. TEMPORAL DISTRIBUTION")
    print("-" * 40)
    if 'email_ts' in df.columns:
        df_sorted = df.sort_values('email_ts')
        print(f"Date range: {df_sorted['email_ts'].min()} to {df_sorted['email_ts'].max()}")
        
        df['month'] = df['email_ts'].dt.to_period('M')
        monthly = df.groupby('month').agg({
            'posterior_prob': ['mean', 'max', 'count']
        }).round(6)
        print("\nMonthly Statistics:")
        print(monthly)
    
    if show_plots:
        print("\n6. GENERATING DIAGNOSTIC PLOTS...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].hist(df['posterior_prob'], bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('Posterior Probability')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Posterior Probabilities')
        axes[0, 0].axvline(df['posterior_prob'].mean(), color='r', linestyle='--', label=f'Mean: {df['posterior_prob'].mean():.6f}')
        axes[0, 0].legend()
        
        axes[0, 1].scatter(df['text_score'], df['posterior_prob'], alpha=0.5)
        axes[0, 1].set_xlabel('Text Score')
        axes[0, 1].set_ylabel('Posterior Probability')
        axes[0, 1].set_title('Text Score vs Posterior')
        
        lr_cols = [c for c in ['lr_text', 'lr_time', 'lr_graph', 'lr_anom'] if c in df.columns]
        lr_means = [df[c].mean() for c in lr_cols]
        axes[0, 2].bar(range(len(lr_cols)), lr_means)
        axes[0, 2].set_xticks(range(len(lr_cols)))
        axes[0, 2].set_xticklabels([c.replace('lr_', '') for c in lr_cols], rotation=45)
        axes[0, 2].set_ylabel('Mean Likelihood Ratio')
        axes[0, 2].set_title('Mean Likelihood Ratios by Feature')
        axes[0, 2].axhline(y=1, color='r', linestyle='--', label='Neutral')
        axes[0, 2].legend()
        
        if 'email_ts' in df.columns:
            df_sorted = df.sort_values('email_ts')
            axes[1, 0].plot(df_sorted['email_ts'], df_sorted['posterior_prob'], 'o', alpha=0.5)
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Posterior Probability')
            axes[1, 0].set_title('Posterior Probability Over Time')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        if 'lexicon_hits' in df.columns:
            axes[1, 1].scatter(df['lexicon_hits'], df['posterior_prob'], alpha=0.5)
            axes[1, 1].set_xlabel('Lexicon Hits')
            axes[1, 1].set_ylabel('Posterior Probability')
            axes[1, 1].set_title('Lexicon Hits vs Posterior')
        
        top_20 = df.nlargest(20, 'posterior_prob')
        axes[1, 2].barh(range(len(top_20)), top_20['posterior_prob'].values)
        axes[1, 2].set_yticks(range(len(top_20)))
        axes[1, 2].set_yticklabels([f"Email {i+1}" for i in range(len(top_20))], fontsize=8)
        axes[1, 2].set_xlabel('Posterior Probability')
        axes[1, 2].set_title('Top 20 Emails by Posterior')
        axes[1, 2].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('bayes_validation_diagnostics.png', dpi=300, bbox_inches='tight')
        print("Saved diagnostic plots to: bayes_validation_diagnostics.png")
        plt.show()
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

# ===== Main Execution =====
if __name__ == "__main__":
    print("Loading BM25 text scores...")
    
    text_scores = pd.read_csv("text_scores_full.csv")
    
    print(f"Loaded {len(text_scores)} emails with text scores")
    print(f"Columns: {list(text_scores.columns)}")
    
    prior = Prior(pi=1e-3)
    params = ChannelParams()
    
    print("\n" + "="*80)
    print("RUNNING BAYESIAN SCORING")
    print("="*80)
    
    scored_emails = score_emails_bayes(
        text_scores, 
        prior, 
        params,
        episode_ts=pd.Timestamp('2001-05-15 09:00:00', tz='UTC')
    )
    
    validate_scoring(scored_emails, show_plots=True)
    
    print("\n" + "="*80)
    print("TOP 10 EMAILS BY BAYESIAN POSTERIOR PROBABILITY")
    print("="*80)
    
    display_cols = ['email_id', 'email_ts', 'posterior_prob', 'text_score', 
                   'lexicon_hits', 'lr_text', 'lr_time']
    
    top_10 = scored_emails.head(10)
    for idx, row in top_10.iterrows():
        print(f"\n{'='*80}")
        print(f"RANK #{idx+1}")
        print(f"{'='*80}")
        print(f"Email ID: {row['email_id']}")
        print(f"Date: {row['email_ts']}")
        print(f"\nSCORES:")
        print(f"  Posterior Probability: {row['posterior_prob']:.8f}")
        print(f"  Text Score: {row['text_score']:.6f}")
        print(f"  Lexicon Hits: {row.get('lexicon_hits', 0)}")
        print(f"\nLIKELIHOOD RATIOS:")
        print(f"  Text LR: {row['lr_text']:.4f}")
        print(f"  Time LR: {row['lr_time']:.4f}")
        print(f"  Graph LR: {row['lr_graph']:.4f}")
        print(f"  Anomaly LR: {row['lr_anom']:.4f}")
        
        if 'subject' in row:
            print(f"\nSubject: {row['subject'][:100]}...")
    
    output_file = "bayes_scored_emails.csv"
    scored_emails.to_csv(output_file, index=False)
    print(f"\n\nSaved Bayesian scores to: {output_file}")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total emails scored: {len(scored_emails)}")
    print(f"Mean posterior probability: {scored_emails['posterior_prob'].mean():.8f}")
    print(f"Median posterior probability: {scored_emails['posterior_prob'].median():.8f}")
    print(f"Emails with posterior > 0.001: {(scored_emails['posterior_prob'] > 0.001).sum()}")
    print(f"Emails with posterior > 0.0001: {(scored_emails['posterior_prob'] > 0.0001).sum()}")