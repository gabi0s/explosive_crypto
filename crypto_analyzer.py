#!/usr/bin/env python3
"""
Crypto Volatility Analyzer - Raspberry Pi 5 Optimized
Analyse la volatilité et identifie les cryptomonnaies prometteuses
"""

import asyncio
import aiohttp
import argparse
import json
import numpy as np
import pandas as pd
import statistics
import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CryptoAnalysis:
    symbol: str
    name: str
    price: float
    volatility_7d: float
    volatility_30d: float
    volume_24h: float
    market_cap: float
    price_change_24h: float
    price_change_7d: float
    price_change_30d: float
    rsi: float
    explosion_score: float
    recommendation: str

class CryptoAnalyzer:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or mp.cpu_count()
        self.session = None
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('crypto_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def get_crypto_list(self, limit=500):
        """Récupère la liste des cryptomonnaies avec leurs données de base"""
        url = f"https://api.coingecko.com/api/v3/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': limit,
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '24h,7d,30d'
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                self.logger.error(f"Erreur API CoinGecko: {response.status}")
                return []

    async def get_historical_data(self, crypto_id, days=30):
        """Récupère les données historiques pour un crypto"""
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily' if days > 90 else 'hourly'
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('prices', [])
                else:
                    return []
        except Exception as e:
            self.logger.error(f"Erreur pour {crypto_id}: {e}")
            return []

    def calculate_volatility(self, prices):
        """Calcule la volatilité (écart-type) des prix"""
        if len(prices) < 2:
            return 0
        
        price_values = [price[1] for price in prices]
        returns = [(price_values[i] - price_values[i-1]) / price_values[i-1] 
                  for i in range(1, len(price_values))]
        
        return statistics.stdev(returns) * 100 if len(returns) > 1 else 0

    def calculate_rsi(self, prices, period=14):
        """Calcule le RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return 50  # Valeur neutre
        
        price_values = [price[1] for price in prices]
        deltas = [price_values[i] - price_values[i-1] for i in range(1, len(price_values))]
        
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def calculate_explosion_score(self, analysis_data):
        """Calcule un score d'explosion basé sur plusieurs critères"""
        score = 0
        
        # Volatilité (30% du score)
        vol_score = min(analysis_data['volatility_7d'] / 10, 10) * 3
        score += vol_score
        
        # Volume relatif au market cap (20% du score)
        if analysis_data['market_cap'] > 0:
            volume_ratio = analysis_data['volume_24h'] / analysis_data['market_cap']
            volume_score = min(volume_ratio * 1000, 10) * 2
            score += volume_score
        
        # Momentum (30% du score)
        momentum = (analysis_data['price_change_7d'] + analysis_data['price_change_30d']) / 2
        momentum_score = min(max(momentum / 10, -10), 10) + 10  # Normalise entre 0 et 20
        score += momentum_score * 1.5
        
        # RSI pour identifier les conditions de survente/surachat (20% du score)
        rsi = analysis_data['rsi']
        if 30 <= rsi <= 70:  # Zone optimale
            rsi_score = 2
        elif rsi < 30:  # Survente - potentiel de rebond
            rsi_score = 1.5
        else:  # Surachat
            rsi_score = 0.5
        score += rsi_score
        
        return min(score, 100)

    def get_recommendation(self, score, volatility, rsi):
        """Génère une recommandation basée sur l'analyse"""
        if score >= 75:
            return "🚀 EXPLOSION POTENTIELLE"
        elif score >= 60:
            return "📈 TRÈS PROMETTEUR"
        elif score >= 45:
            return "💡 PROMETTEUR"
        elif score >= 30:
            return "⚖️ NEUTRE"
        else:
            return "🔴 ÉVITER"

    async def analyze_crypto(self, crypto_data):
        """Analyse une cryptomonnaie spécifique"""
        crypto_id = crypto_data['id']
        
        # Données historiques 7j et 30j
        hist_7d = await self.get_historical_data(crypto_id, 7)
        hist_30d = await self.get_historical_data(crypto_id, 30)
        
        await asyncio.sleep(0.1)  # Rate limiting
        
        if not hist_7d or not hist_30d:
            return None
        
        vol_7d = self.calculate_volatility(hist_7d)
        vol_30d = self.calculate_volatility(hist_30d)
        rsi = self.calculate_rsi(hist_30d)
        
        analysis_data = {
            'volatility_7d': vol_7d,
            'volatility_30d': vol_30d,
            'volume_24h': crypto_data.get('total_volume', 0),
            'market_cap': crypto_data.get('market_cap', 0),
            'price_change_7d': crypto_data.get('price_change_percentage_7d_in_currency', 0) or 0,
            'price_change_30d': crypto_data.get('price_change_percentage_30d_in_currency', 0) or 0,
            'rsi': rsi
        }
        
        explosion_score = self.calculate_explosion_score(analysis_data)
        recommendation = self.get_recommendation(explosion_score, vol_7d, rsi)
        
        return CryptoAnalysis(
            symbol=crypto_data['symbol'].upper(),
            name=crypto_data['name'],
            price=crypto_data['current_price'],
            volatility_7d=vol_7d,
            volatility_30d=vol_30d,
            volume_24h=analysis_data['volume_24h'],
            market_cap=analysis_data['market_cap'],
            price_change_24h=crypto_data.get('price_change_percentage_24h_in_currency', 0) or 0,
            price_change_7d=analysis_data['price_change_7d'],
            price_change_30d=analysis_data['price_change_30d'],
            rsi=rsi,
            explosion_score=explosion_score,
            recommendation=recommendation
        )

    async def analyze_all_cryptos(self, limit=200, min_market_cap=1000000):
        """Analyse toutes les cryptomonnaies"""
        self.logger.info(f"🔄 Récupération des données pour {limit} cryptomonnaies...")
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            cryptos = await self.get_crypto_list(limit)
            if not cryptos:
                self.logger.error("❌ Impossible de récupérer les données")
                return []
            
            # Filtrer par market cap minimum
            filtered_cryptos = [
                crypto for crypto in cryptos 
                if crypto.get('market_cap', 0) >= min_market_cap
            ]
            
            self.logger.info(f"📊 Analyse de {len(filtered_cryptos)} cryptomonnaies...")
            
            # Analyse en parallèle avec limitation pour éviter le rate limiting
            semaphore = asyncio.Semaphore(5)  # Max 5 requêtes simultanées
            
            async def analyze_with_semaphore(crypto):
                async with semaphore:
                    return await self.analyze_crypto(crypto)
            
            tasks = [analyze_with_semaphore(crypto) for crypto in filtered_cryptos]
            results = []
            
            for i, task in enumerate(asyncio.as_completed(tasks)):
                result = await task
                if result:
                    results.append(result)
                if (i + 1) % 10 == 0:
                    self.logger.info(f"✅ {i + 1}/{len(filtered_cryptos)} cryptos analysées")
            
            return results

    def display_results(self, results, sort_by='explosion_score', limit=50, min_score=0):
        """Affiche les résultats de l'analyse"""
        if not results:
            print("❌ Aucun résultat à afficher")
            return
        
        # Filtrer et trier
        filtered_results = [r for r in results if r.explosion_score >= min_score]
        sorted_results = sorted(filtered_results, 
                              key=lambda x: getattr(x, sort_by), 
                              reverse=True)[:limit]
        
        print(f"\n🎯 TOP {len(sorted_results)} CRYPTOMONNAIES - Triées par {sort_by.upper()}")
        print("=" * 120)
        
        # En-tête
        print(f"{'RANG':<4} {'SYMBOLE':<8} {'NOM':<20} {'PRIX $':<12} {'VOL 7J%':<8} {'VOL 30J%':<9} "
              f"{'RSI':<6} {'SCORE':<6} {'24H%':<8} {'7J%':<8} {'RECOMMANDATION'}")
        print("-" * 120)
        
        # Résultats
        for i, crypto in enumerate(sorted_results, 1):
            print(f"{i:<4} {crypto.symbol:<8} {crypto.name[:18]:<20} "
                  f"${crypto.price:<11.6f} {crypto.volatility_7d:<7.2f}% {crypto.volatility_30d:<8.2f}% "
                  f"{crypto.rsi:<5.1f} {crypto.explosion_score:<5.1f} "
                  f"{crypto.price_change_24h:<7.2f}% {crypto.price_change_7d:<7.2f}% "
                  f"{crypto.recommendation}")
        
        print("\n📈 STATISTIQUES GLOBALES:")
        print(f"   • Moyenne volatilité 7j: {np.mean([r.volatility_7d for r in sorted_results]):.2f}%")
        print(f"   • Moyenne score explosion: {np.mean([r.explosion_score for r in sorted_results]):.1f}/100")
        print(f"   • Cryptos avec score > 60: {len([r for r in sorted_results if r.explosion_score > 60])}")

    def export_results(self, results, filename=None):
        """Exporte les résultats en CSV"""
        if not filename:
            filename = f"crypto_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        data = []
        for crypto in results:
            data.append({
                'Symbol': crypto.symbol,
                'Name': crypto.name,
                'Price': crypto.price,
                'Volatility_7d': crypto.volatility_7d,
                'Volatility_30d': crypto.volatility_30d,
                'RSI': crypto.rsi,
                'Explosion_Score': crypto.explosion_score,
                'Price_Change_24h': crypto.price_change_24h,
                'Price_Change_7d': crypto.price_change_7d,
                'Price_Change_30d': crypto.price_change_30d,
                'Volume_24h': crypto.volume_24h,
                'Market_Cap': crypto.market_cap,
                'Recommendation': crypto.recommendation
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"📁 Résultats exportés dans {filename}")

async def main():
    parser = argparse.ArgumentParser(description='Crypto Volatility Analyzer pour Raspberry Pi 5')
    parser.add_argument('--limit', type=int, default=200, help='Nombre de cryptos à analyser (défaut: 200)')
    parser.add_argument('--min-market-cap', type=int, default=1000000, help='Market cap minimum (défaut: 1M$)')
    parser.add_argument('--sort-by', choices=['explosion_score', 'volatility_7d', 'volatility_30d', 'price_change_7d'], 
                       default='explosion_score', help='Critère de tri (défaut: explosion_score)')
    parser.add_argument('--top', type=int, default=50, help='Nombre de résultats à afficher (défaut: 50)')
    parser.add_argument('--min-score', type=float, default=0, help='Score minimum pour filtrer (défaut: 0)')
    parser.add_argument('--export', action='store_true', help='Exporter les résultats en CSV')
    parser.add_argument('--export-file', type=str, help='Nom du fichier d\'export personnalisé')
    
    args = parser.parse_args()
    
    print("🚀 CRYPTO VOLATILITY ANALYZER - RASPBERRY PI 5")
    print("=" * 60)
    
    analyzer = CryptoAnalyzer()
    
    start_time = time.time()
    results = await analyzer.analyze_all_cryptos(args.limit, args.min_market_cap)
    end_time = time.time()
    
    if results:
        print(f"\n⏱️  Analyse terminée en {end_time - start_time:.2f} secondes")
        analyzer.display_results(results, args.sort_by, args.top, args.min_score)
        
        if args.export:
            analyzer.export_results(results, args.export_file)
    else:
        print("❌ Aucune donnée analysée")

if __name__ == "__main__":
    asyncio.run(main())