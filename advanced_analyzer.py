#!/usr/bin/env python3
"""
Module d'analyse avancée pour le Crypto Volatility Analyzer
Version corrigée avec gestion robuste des types et erreurs
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, Optional
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta

class AdvancedCryptoAnalyzer:
    def __init__(self):
        self.session = None
        self.request_timeout = aiohttp.ClientTimeout(total=30)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    def _sanitize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Union[float, int, str]]:
        """Convertit les valeurs booléennes et None en types valides"""
        sanitized = {}
        for k, v in metrics.items():
            if isinstance(v, bool):
                sanitized[k] = int(v)
            elif v is None:
                sanitized[k] = 0
            elif isinstance(v, (str, int, float)):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized

    async def get_defi_metrics(self, crypto_id: str) -> Dict[str, Union[float, int, str]]:
        """Récupère les métriques DeFi spécifiques"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}"
            async with self.session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._sanitize_metrics({
                        'total_value_locked': data.get('market_data', {}).get('total_value_locked', {}),
                        'developer_score': data.get('developer_score', 0),
                        'community_score': data.get('community_score', 0),
                        'liquidity_score': data.get('liquidity_score', 0),
                        'public_interest_score': data.get('public_interest_score', 0)
                    })
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            print(f"Erreur get_defi_metrics pour {crypto_id}: {str(e)}")
        return {}

    def calculate_bollinger_bands(self, prices: List[Tuple[float, float]], period: int = 20, std_dev: int = 2
                                ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calcule les bandes de Bollinger"""
        if len(prices) < period:
            return None, None, None
            
        try:
            price_values = [p[1] for p in prices[-period:]]
            sma = np.mean(price_values)
            std = np.std(price_values)
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return upper_band, sma, lower_band
        except Exception as e:
            print(f"Erreur calculate_bollinger_bands: {str(e)}")
            return None, None, None

    def calculate_macd(self, prices: List[Tuple[float, float]], fast: int = 12, slow: int = 26, signal: int = 9
                      ) -> Tuple[float, float, float]:
        """Calcule le MACD"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
            
        try:
            price_values = [p[1] for p in prices]
            
            def calculate_ema(values: List[float], period: int) -> List[float]:
                multiplier = 2 / (period + 1)
                ema = [values[0]]
                for i in range(1, len(values)):
                    ema.append((values[i] * multiplier) + (ema[-1] * (1 - multiplier)))
                return ema
            
            ema_fast = calculate_ema(price_values, fast)
            ema_slow = calculate_ema(price_values, slow)
            
            if len(ema_fast) < slow or len(ema_slow) < slow:
                return 0.0, 0.0, 0.0
                
            macd_line = [ema_fast[i] - ema_slow[i] for i in range(slow-1, len(ema_fast))]
            
            if len(macd_line) < signal:
                return 0.0, 0.0, 0.0
                
            signal_line = calculate_ema(macd_line, signal)
            histogram = [macd_line[i] - signal_line[i] for i in range(len(signal_line))]
            
            return (
                macd_line[-1] if macd_line else 0.0,
                signal_line[-1] if signal_line else 0.0,
                histogram[-1] if histogram else 0.0
            )
        except Exception as e:
            print(f"Erreur calculate_macd: {str(e)}")
            return 0.0, 0.0, 0.0

    def calculate_stochastic_oscillator(self, prices: List[Tuple[float, ...]], k_period: int = 14, d_period: int = 3
                                      ) -> Tuple[float, float]:
        """Calcule l'oscillateur stochastique"""
        if len(prices) < k_period:
            return 50.0, 50.0
            
        try:
            highs = [p[2] if len(p) > 2 else p[1] for p in prices[-k_period:]]
            lows = [p[3] if len(p) > 3 else p[1] for p in prices[-k_period:]]
            closes = [p[1] for p in prices[-k_period:]]
            
            highest_high = max(highs)
            lowest_low = min(lows)
            current_close = closes[-1]
            
            if highest_high == lowest_low:
                k_percent = 50.0
            else:
                k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
            
            d_percent = k_percent  # Simplification
            
            return k_percent, d_percent
        except Exception as e:
            print(f"Erreur calculate_stochastic_oscillator: {str(e)}")
            return 50.0, 50.0

    def detect_patterns(self, prices: List[Tuple[float, float]]) -> List[str]:
        """Détecte des patterns techniques"""
        if len(prices) < 10:
            return []
            
        try:
            price_values = [p[1] for p in prices[-10:]]
            patterns = []
            
            # Double Bottom
            if len(price_values) >= 5:
                min1 = min(price_values[:3])
                min2 = min(price_values[-3:])
                if abs(min1 - min2) / min1 < 0.02:
                    patterns.append("double_bottom")
            
            # Trend
            recent_trend = np.polyfit(range(5), price_values[-5:], 1)[0]
            if recent_trend > 0:
                patterns.append("uptrend")
            elif recent_trend < 0:
                patterns.append("downtrend")
                
            # Breakout
            recent_high = max(price_values[-3:])
            previous_high = max(price_values[-10:-3])
            if recent_high > previous_high * 1.05:
                patterns.append("breakout")
                
            return patterns
        except Exception as e:
            print(f"Erreur detect_patterns: {str(e)}")
            return []

    def calculate_support_resistance(self, prices: List[Tuple[float, float]]
                                   ) -> Tuple[Optional[float], Optional[float]]:
        """Calcule les niveaux de support et résistance"""
        if len(prices) < 20:
            return None, None
            
        try:
            price_values = [p[1] for p in prices]
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
        except Exception as e:
            print(f"Erreur calculate_support_resistance: {str(e)}")
            return None, None

    def calculate_volume_profile(self, prices: List[Tuple[float, float]], 
                                volumes: List[Tuple[float, float]]) -> Dict[str, float]:
        """Analyse le profil de volume"""
        if len(prices) != len(volumes) or len(prices) < 10:
            return {}
            
        try:
            price_volume = list(zip([p[1] for p in prices], [v[1] for v in volumes]))
            price_zones = {}
            
            for price, volume in price_volume:
                zone = round(price, -int(np.log10(price)) + 1)
                price_zones[zone] = price_zones.get(zone, 0) + volume
                
            poc = max(price_zones.items(), key=lambda x: x[1]) if price_zones else (0.0, 0.0)
            
            return {
                'poc_price': float(poc[0]),
                'poc_volume': float(poc[1]),
                'total_zones': len(price_zones),
                'avg_volume_per_zone': float(np.mean(list(price_zones.values())) if price_zones else 0)
            }
        except Exception as e:
            print(f"Erreur calculate_volume_profile: {str(e)}")
            return {}

    async def analyze_social_sentiment(self, crypto_id: str) -> Dict[str, Union[float, int, str]]:
        """Analyse le sentiment social"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/tickers"
            async with self.session.get(url, timeout=self.request_timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    tickers = data.get('tickers', [])
                    exchanges = set()
                    
                    for ticker in tickers:
                        market = ticker.get('market', {})
                        if isinstance(market, dict):
                            exchanges.add(market.get('name', ''))
                    
                    diversity_score = min(len(exchanges) * 10, 100)
                    
                    return self._sanitize_metrics({
                        'exchange_diversity': diversity_score,
                        'total_exchanges': len(exchanges),
                        'total_pairs': len(tickers)
                    })
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            print(f"Erreur analyze_social_sentiment pour {crypto_id}: {str(e)}")
        
        return self._sanitize_metrics({
            'exchange_diversity': 50,
            'total_exchanges': 1,
            'total_pairs': 1
        })

    def calculate_sharpe_ratio(self, prices: List[Tuple[float, float]], risk_free_rate: float = 0.02) -> float:
        """Calcule le ratio de Sharpe"""
        if len(prices) < 30:
            return 0.0
            
        try:
            price_values = [p[1] for p in prices]
            returns = [(price_values[i] - price_values[i-1]) / price_values[i-1] 
                      for i in range(1, len(price_values))]
            
            if not returns:
                return 0.0
                
            avg_return = np.mean(returns) * 365
            volatility = np.std(returns) * np.sqrt(365)
            
            if volatility == 0:
                return 0.0
                
            return float((avg_return - risk_free_rate) / volatility)
        except Exception as e:
            print(f"Erreur calculate_sharpe_ratio: {str(e)}")
            return 0.0

    def calculate_maximum_drawdown(self, prices: List[Tuple[float, float]]) -> float:
        """Calcule le maximum drawdown"""
        if len(prices) < 2:
            return 0.0
            
        try:
            price_values = [p[1] for p in prices]
            peak = price_values[0]
            max_drawdown = 0.0
            
            for price in price_values[1:]:
                if price > peak:
                    peak = price
                drawdown = (peak - price) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    
            return float(max_drawdown * 100)
        except Exception as e:
            print(f"Erreur calculate_maximum_drawdown: {str(e)}")
            return 0.0

    async def get_advanced_metrics(self, crypto_id: str, 
                                 prices: List[Tuple[float, float]], 
                                 volumes: Optional[List[Tuple[float, float]]] = None
                                ) -> Dict[str, Union[float, int, str, List[str]]]:
        """Calcule toutes les métriques avancées"""
        metrics = {}
        
        try:
            # Indicateurs techniques
            upper_band, middle_band, lower_band = self.calculate_bollinger_bands(prices)
            macd, signal, histogram = self.calculate_macd(prices)
            stoch_k, stoch_d = self.calculate_stochastic_oscillator(prices)
            
            metrics.update(self._sanitize_metrics({
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
            }))
            
            # Support/Résistance
            support, resistance = self.calculate_support_resistance(prices)
            metrics.update(self._sanitize_metrics({
                'support_level': support,
                'resistance_level': resistance
            }))
            
            # Patterns
            metrics['patterns'] = self.detect_patterns(prices)
            
            # Volume profile
            if volumes:
                metrics.update(self._sanitize_metrics(
                    self.calculate_volume_profile(prices, volumes)
                ))
            
            # Sentiment social
            metrics.update(await self.analyze_social_sentiment(crypto_id))
            
            # Position Bollinger
            current_price = prices[-1][1] if prices else 0.0
            if all([upper_band, lower_band, middle_band]):
                if current_price > upper_band:
                    metrics['bollinger_position'] = 'overbought'
                elif current_price < lower_band:
                    metrics['bollinger_position'] = 'oversold'
                else:
                    metrics['bollinger_position'] = 'normal'
                    
            return metrics
            
        except Exception as e:
            print(f"Erreur critique get_advanced_metrics pour {crypto_id}: {str(e)}")
            return self._sanitize_metrics(metrics)

    def calculate_advanced_explosion_score(self, basic_analysis: Any, 
                                         advanced_metrics: Dict[str, Any]) -> float:
        """Calcule un score d'explosion amélioré"""
        try:
            score = float(getattr(basic_analysis, 'explosion_score', 50))
            
            # MACD
            if advanced_metrics.get('macd', 0) > advanced_metrics.get('macd_signal', 0):
                score += 5
            else:
                score -= 2
                
            # Bollinger
            bollinger_pos = advanced_metrics.get('bollinger_position', 'normal')
            if bollinger_pos == 'oversold':
                score += 8
            elif bollinger_pos == 'overbought':
                score -= 5
                
            # Stochastic
            stoch_k = advanced_metrics.get('stochastic_k', 50)
            if stoch_k < 20:
                score += 6
            elif stoch_k > 80:
                score -= 4
                
            # Patterns
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
                
            # Drawdown
            max_dd = advanced_metrics.get('max_drawdown', 0)
            if max_dd > 50:
                score -= 8
            elif max_dd < 20:
                score += 3
                
            # Exchange Diversity
            exchange_diversity = advanced_metrics.get('exchange_diversity', 50)
            if exchange_diversity > 80:
                score += 4
            elif exchange_diversity < 30:
                score -= 3
                
            return max(0.0, min(score, 100.0))
            
        except Exception as e:
            print(f"Erreur calculate_advanced_explosion_score: {str(e)}")
            return 50.0

    def get_basic_recommendation(self, score: float) -> str:
        """Recommandation de base selon le score"""
        try:
            score = float(score)
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
            return "🔴 ÉVITER"
        except:
            return "⚖️ NEUTRE"

    def get_advanced_recommendation(self, score: float, 
                                  advanced_metrics: Dict[str, Any]) -> str:
        """Génère une recommandation détaillée"""
        base_rec = self.get_basic_recommendation(score)
        context = []
        
        try:
            patterns = advanced_metrics.get('patterns', [])
            bollinger_pos = advanced_metrics.get('bollinger_position', 'normal')
            
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
        except Exception as e:
            print(f"Erreur get_advanced_recommendation: {str(e)}")
            
        return base_rec

async def example_usage():
    """Exemple d'utilisation de l'analyseur avancé"""
    async with AdvancedCryptoAnalyzer() as analyzer:
        # Données exemple (timestamp, price, high, low)
        prices = [
            (datetime(2023,1,i).timestamp(), i*100, i*100+5, i*100-5) 
            for i in range(1, 31)
        ]
        
        # Volumes exemple (timestamp, volume)
        volumes = [
            (datetime(2023,1,i).timestamp(), i*10000) 
            for i in range(1, 31)
        ]
        
        metrics = await analyzer.get_advanced_metrics("bitcoin", prices, volumes)
        print("Métriques avancées:", json.dumps(metrics, indent=2))
        
        class BasicAnalysis:
            explosion_score = 65
        
        advanced_score = analyzer.calculate_advanced_explosion_score(
            BasicAnalysis(), metrics
        )
        print("Score avancé:", advanced_score)
        
        recommendation = analyzer.get_advanced_recommendation(
            advanced_score, metrics
        )
        print("Recommandation:", recommendation)

if __name__ == "__main__":
    asyncio.run(example_usage())