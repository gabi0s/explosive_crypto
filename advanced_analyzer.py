#!/usr/bin/env python3
"""
Module d'analyse avancée pour le Crypto Volatility Analyzer
Inclut des algorithmes sophistiqués pour identifier les cryptos prometteuses
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta

class AdvancedCryptoAnalyzer:
    def __init__(self):
        self.session = None
        
    async def get_defi_metrics(self, crypto_id):
        """Récupère les métriques DeFi spécifiques"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'total_value_locked': data.get('market_data', {}).get('total_value_locked', {}),
                        'developer_score': data.get('developer_score', 0),
                        'community_score': data.get('community_score', 0),
                        'liquidity_score': data.get('liquidity_score', 0),
                        'public_interest_score': data.get('public_interest_score', 0)
                    }
        except:
            pass
        return {}

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calcule les bandes de Bollinger"""
        if len(prices) < period:
            return None, None, None
            
        price_values = [p[1] for p in prices[-period:]]
        sma = np.mean(price_values)
        std = np.std(price_values)
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcule le MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return 0, 0, 0
            
        price_values = [p[1] for p in prices]
        
        # EMA calculation
        def calculate_ema(values, period):
            multiplier = 2 / (period + 1)
            ema = [values[0]]
            for i in range(1, len(values)):
                ema.append((values[i] * multiplier) + (ema[-1] * (1 - multiplier)))
            return ema
        
        ema_fast = calculate_ema(price_values, fast)
        ema_slow = calculate_ema(price_values, slow)
        
        if len(ema_fast) < slow or len(ema_slow) < slow:
            return 0, 0, 0
            
        macd_line = [ema_fast[i] - ema_slow[i] for i in range(slow-1, len(ema_fast))]
        
        if len(macd_line) < signal:
            return 0, 0, 0
            
        signal_line = calculate_ema(macd_line, signal)
        histogram = [macd_line[i] - signal_line[i] for i in range(len(signal_line))]
        
        return macd_line[-1] if macd_line else 0, signal_line[-1] if signal_line else 0, histogram[-1] if histogram else 0

    def calculate_stochastic_oscillator(self, prices, k_period=14, d_period=3):
        """Calcule l'oscillateur stochastique"""
        if len(prices) < k_period:
            return 50, 50
            
        highs = [p[2] if len(p) > 2 else p[1] for p in prices[-k_period:]]
        lows = [p[3] if len(p) > 3 else p[1] for p in prices[-k_period:]]
        closes = [p[1] for p in prices[-k_period:]]
        
        highest_high = max(highs)
        lowest_low = min(lows)
        current_close = closes[-1]
        
        if highest_high == lowest_low:
            k_percent = 50
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # D% est la moyenne mobile de K%
        d_percent = k_percent  # Simplification pour cet exemple
        
        return k_percent, d_percent

    def detect_patterns(self, prices):
        """Détecte des patterns techniques simples"""
        if len(prices) < 10:
            return []
            
        price_values = [p[1] for p in prices[-10:]]
        patterns = []
        
        # Double Bottom
        if len(price_values) >= 5:
            min1 = min(price_values[:3])
            min2 = min(price_values[-3:])
            if abs(min1 - min2) / min1 < 0.02:  # 2% de tolérance
                patterns.append("double_bottom")
        
        # Trend ascendant
        recent_trend = np.polyfit(range(5), price_values[-5:], 1)[0]
        if recent_trend > 0:
            patterns.append("uptrend")
        elif recent_trend < 0:
            patterns.append("downtrend")
            
        # Breakout detection
        recent_high = max(price_values[-3:])
        previous_high = max(price_values[-10:-3])
        if recent_high > previous_high * 1.05:  # 5% breakout
            patterns.append("breakout")
            
        return patterns

    def calculate_support_resistance(self, prices):
        """Calcule les niveaux de support et résistance"""
        if len(prices) < 20:
            return None, None
            
        price_values = [p[1] for p in prices]
        
        # Méthode simplifiée basée sur les pivots
        highs = []
        lows = []
        
        for i in range(2, len(price_values) - 2):
            if (price_values[i] > price_values[i-1] and 
                price_values[i] > price_values[i+1] and
                price_values[i] > price_values[i-2] and 
                price_values[i] > price_values[i+2]):
                highs.append(price_values[i])
                
            if (price_values[i] < price_values[i-1] and 
                price_values[i] < price_values[i+1] and
                price_values[i] < price_values[i-2] and 
                price_values[i] < price_values[i+2]):
                lows.append(price_values[i])
        
        resistance = np.mean(highs) if highs else None
        support = np.mean(lows) if lows else None
        
        return support, resistance

    def calculate_volume_profile(self, prices, volumes):
        """Analyse le profil de volume"""
        if len(prices) != len(volumes) or len(prices) < 10:
            return {}
            
        price_volume = list(zip([p[1] for p in prices], [v[1] for v in volumes]))
        
        # Grouper par zones de prix
        price_zones = {}
        for price, volume in price_volume:
            zone = round(price, -int(np.log10(price)) + 1)  # Arrondi intelligent
            if zone not in price_zones:
                price_zones[zone] = 0
            price_zones[zone] += volume
            
        # Trouver la zone de plus fort volume (POC - Point of Control)
        poc = max(price_zones.items(), key=lambda x: x[1]) if price_zones else (0, 0)
        
        return {
            'poc_price': poc[0],
            'poc_volume': poc[1],
            'total_zones': len(price_zones),
            'avg_volume_per_zone': np.mean(list(price_zones.values()))
        }

    async def analyze_social_sentiment(self, crypto_id):
        """Analyse le sentiment social (APIs externes)"""
        # Cette fonction nécessiterait des APIs comme Reddit, Twitter, etc.
        # Pour l'instant, on simule avec des données CoinGecko
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/tickers"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    tickers = data.get('tickers', [])
                    
                    # Calcul de score basé sur la diversité des exchanges
                    exchanges = set([ticker.get('market', {}).get('name', '') for ticker in tickers])
                    diversity_score = min(len(exchanges) * 10, 100)
                    
                    return {
                        'exchange_diversity': diversity_score,
                        'total_exchanges': len(exchanges),
                        'total_pairs': len(tickers)
                    }
        except:
            pass
            
        return {'exchange_diversity': 50, 'total_exchanges': 1, 'total_pairs': 1}

    def calculate_sharpe_ratio(self, prices, risk_free_rate=0.02):
        """Calcule le ratio de Sharpe"""
        if len(prices) < 30:
            return 0
            
        price_values = [p[1] for p in prices]
        returns = [(price_values[i] - price_values[i-1]) / price_values[i-1] 
                  for i in range(1, len(price_values))]
        
        if not returns:
            return 0
            
        avg_return = np.mean(returns) * 365  # Annualisé
        volatility = np.std(returns) * np.sqrt(365)  # Annualisé
        
        if volatility == 0:
            return 0
            
        sharpe = (avg_return - risk_free_rate) / volatility
        return sharpe

    def calculate_maximum_drawdown(self, prices):
        """Calcule le maximum drawdown"""
        if len(prices) < 2:
            return 0
            
        price_values = [p[1] for p in prices]
        peak = price_values[0]
        max_drawdown = 0
        
        for price in price_values[1:]:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        return max_drawdown * 100

    async def get_advanced_metrics(self, crypto_id, prices, volumes=None):
        """Calcule toutes les métriques avancées"""
        metrics = {}
        
        # Indicateurs techniques
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(prices)
        macd, signal, histogram = self.calculate_macd(prices)
        stoch_k, stoch_d = self.calculate_stochastic_oscillator(prices)
        
        metrics.update({
            'bollinger_upper': upper_band,
            'bollinger_middle': middle_band,
            'bollinger_lower': lower_band,
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': histogram,
            'stochastic_k': stoch_k,
            'stochastic_d': stoch_d,
            'sharpe_ratio': self.calculate_sharpe_ratio(prices),
            'max_drawdown': self.calculate_maximum_drawdown(prices)
        })
        
        # Support/Résistance
        support, resistance = self.calculate_support_resistance(prices)
        metrics['support_level'] = support
        metrics['resistance_level'] = resistance
        
        # Patterns
        metrics['patterns'] = self.detect_patterns(prices)
        
        # Volume profile si disponible
        if volumes:
            volume_profile = self.calculate_volume_profile(prices, volumes)
            metrics.update(volume_profile)
        
        # Sentiment social
        social_metrics = await self.analyze_social_sentiment(crypto_id)
        metrics.update(social_metrics)
        
        # Position par rapport aux bandes de Bollinger
        current_price = prices[-1][1] if prices else 0
        if upper_band and lower_band and middle_band:
            if current_price > upper_band:
                metrics['bollinger_position'] = 'overbought'
            elif current_price < lower_band:
                metrics['bollinger_position'] = 'oversold'
            else:
                metrics['bollinger_position'] = 'normal'
        for k, v in metrics.items():
            if isinstance(v, bool):
                metrics[k] = int(v)
            elif v is None:
                metrics[k] = 0
        return metrics

    def calculate_advanced_explosion_score(self, basic_analysis, advanced_metrics):
        """Calcule un score d'explosion amélioré avec les métriques avancées"""
        score = basic_analysis.explosion_score  # Score de base
        
        # Bonus/Malus basés sur les indicateurs techniques
        
        # MACD
        if advanced_metrics.get('macd', 0) > advanced_metrics.get('macd_signal', 0):
            score += 5  # Signal haussier
        else:
            score -= 2  # Signal baissier
            
        # Position Bollinger
        bollinger_pos = advanced_metrics.get('bollinger_position', 'normal')
        if bollinger_pos == 'oversold':
            score += 8  # Potentiel de rebond
        elif bollinger_pos == 'overbought':
            score -= 5  # Risque de correction
            
        # Stochastic
        stoch_k = advanced_metrics.get('stochastic_k', 50)
        if stoch_k < 20:  # Survente
            score += 6
        elif stoch_k > 80:  # Surachat
            score -= 4
            
        # Patterns techniques
        patterns = advanced_metrics.get('patterns', [])
        pattern_bonus = {
            'double_bottom': 10,
            'uptrend': 7,
            'breakout': 15,
            'downtrend': -8
        }
        
        for pattern in patterns:
            score += pattern_bonus.get(pattern, 0)
            
        # Sharpe Ratio
        sharpe = advanced_metrics.get('sharpe_ratio', 0)
        if sharpe > 1:
            score += 5
        elif sharpe < -0.5:
            score -= 5
            
        # Drawdown maximum
        max_dd = advanced_metrics.get('max_drawdown', 0)
        if max_dd > 50:  # Drawdown élevé = risque
            score -= 8
        elif max_dd < 20:  # Drawdown faible = stabilité
            score += 3
            
        # Diversité des exchanges
        exchange_diversity = advanced_metrics.get('exchange_diversity', 50)
        if exchange_diversity > 80:
            score += 4
        elif exchange_diversity < 30:
            score -= 3
            
        return min(max(score, 0), 100)  # Garde le score entre 0 et 100

    def get_advanced_recommendation(self, score, advanced_metrics):
        """Génère une recommandation détaillée"""
        base_rec = self.get_basic_recommendation(score)
        
        patterns = advanced_metrics.get('patterns', [])
        bollinger_pos = advanced_metrics.get('bollinger_position', 'normal')
        
        # Ajouts contextuels
        context = []
        
        if 'breakout' in patterns:
            context.append("📈 BREAKOUT")
        if 'double_bottom' in patterns:
            context.append("🔄 DOUBLE BOTTOM")
        if bollinger_pos == 'oversold':
            context.append("💎 SURVENTE")
        if advanced_metrics.get('macd', 0) > advanced_metrics.get('macd_signal', 0):
            context.append("🚀 MACD+")
            
        if context:
            return f"{base_rec} {' '.join(context)}"
        return base_rec
        
    def get_basic_recommendation(self, score):
        """Recommandation de base selon le score"""
        if score >= 80:
            return "🚀 EXPLOSION IMMINENTE"
        elif score >= 70:
            return "🔥 TRÈS FORTE OPPORTUNITÉ"
        elif score >= 60:
            return "📈 FORTE OPPORTUNITÉ"
        elif score >= 45:
            return "💡 OPPORTUNITÉ MODÉRÉE"
        elif score >= 30:
            return "⚖️ NEUTRE"
        else:
            return "🔴 ÉVITER"