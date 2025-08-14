#!/usr/bin/env python3
"""
Crypto Launcher - Interface principale pour l'analyseur de cryptomonnaies
Intègre l'analyse de base et avancée
"""

import asyncio
import argparse
import sys
import time
from datetime import datetime
import aiohttp

# Import des modules
from crypto_analyzer import CryptoAnalyzer, CryptoAnalysis
from advanced_analyzer import AdvancedCryptoAnalyzer

class CryptoLauncher:
    def __init__(self):
        self.basic_analyzer = CryptoAnalyzer()
        self.advanced_analyzer = AdvancedCryptoAnalyzer()
        
    async def run_basic_analysis(self, args):
        """Lance l'analyse de base"""
        print("🔍 ANALYSE STANDARD")
        print("=" * 50)
        
        results = await self.basic_analyzer.analyze_all_cryptos(
            args.limit, 
            args.min_market_cap
        )
        
        if results:
            self.basic_analyzer.display_results(
                results, 
                args.sort_by, 
                args.top, 
                args.min_score
            )
            
            if args.export:
                self.basic_analyzer.export_results(results, args.export_file)
                
        return results
        
    async def run_advanced_analysis(self, args):
        """Lance l'analyse avancée"""
        print("🚀 ANALYSE AVANCÉE")
        print("=" * 50)
        
        # D'abord récupérer l'analyse de base
        basic_results = await self.basic_analyzer.analyze_all_cryptos(
            args.limit,
            args.min_market_cap
        )
        
        if not basic_results:
            print("❌ Impossible de récupérer les données de base")
            return []
            
        print(f"🔬 Analyse technique avancée sur {len(basic_results)} cryptos...")
        
        enhanced_results = []
        
        async with aiohttp.ClientSession() as session:
            self.advanced_analyzer.session = session
            
            for i, crypto in enumerate(basic_results):
                try:
                    # Récupérer données historiques étendues
                    hist_data = await self.basic_analyzer.get_historical_data(
                        crypto.symbol.lower(), 90
                    )
                    
                    if hist_data and len(hist_data) > 20:
                        # Calculer métriques avancées
                        advanced_metrics = await self.advanced_analyzer.get_advanced_metrics(
                            crypto.symbol.lower(),
                            hist_data
                        )
                        
                        # Recalculer le score avec les métriques avancées
                        enhanced_score = self.advanced_analyzer.calculate_advanced_explosion_score(
                            crypto,
                            advanced_metrics
                        )
                        
                        # Nouvelle recommandation
                        new_recommendation = self.advanced_analyzer.get_advanced_recommendation(
                            enhanced_score,
                            advanced_metrics
                        )
                        
                        # Créer crypto analysis améliorée
                        enhanced_crypto = CryptoAnalysis(
                            symbol=crypto.symbol,
                            name=crypto.name,
                            price=crypto.price,
                            volatility_7d=crypto.volatility_7d,
                            volatility_30d=crypto.volatility_30d,
                            volume_24h=crypto.volume_24h,
                            market_cap=crypto.market_cap,
                            price_change_24h=crypto.price_change_24h,
                            price_change_7d=crypto.price_change_7d,
                            price_change_30d=crypto.price_change_30d,
                            rsi=crypto.rsi,
                            explosion_score=enhanced_score,
                            recommendation=new_recommendation
                        )
                        
                        # Ajouter métriques avancées comme attributs
                        enhanced_crypto.advanced_metrics = advanced_metrics
                        enhanced_results.append(enhanced_crypto)
                        
                    if (i + 1) % 10 == 0:
                        print(f"✅ {i + 1}/{len(basic_results)} analyses avancées terminées")
                        
                    await asyncio.sleep(0.2)  # Rate limiting plus conservateur
                    
                except Exception as e:
                    print(f"⚠️  Erreur pour {crypto.symbol}: {e}")
                    enhanced_results.append(crypto)  # Garder l'analyse de base
                    
        self.display_advanced_results(enhanced_results, args)
        
        if args.export:
            self.export_advanced_results(enhanced_results, args.export_file)
            
        return enhanced_results
        
    def display_advanced_results(self, results, args):
        """Affiche les résultats de l'analyse avancée"""
        if not results:
            print("❌ Aucun résultat à afficher")
            return
            
        # Filtrer et trier
        filtered_results = [r for r in results if r.explosion_score >= args.min_score]
        sorted_results = sorted(filtered_results,
                              key=lambda x: getattr(x, args.sort_by),
                              reverse=True)[:args.top]
        
        print(f"\n🎯 TOP {len(sorted_results)} CRYPTOMONNAIES - ANALYSE AVANCÉE")
        print("=" * 140)
        
        # En-tête étendu
        print(f"{'RANG':<4} {'SYMBOLE':<8} {'NOM':<18} {'PRIX $':<12} {'SCORE':<6} "
              f"{'VOL7J%':<7} {'RSI':<5} {'MACD':<6} {'STOCH':<6} {'PATTERNS':<15} {'RECOMMANDATION'}")
        print("-" * 140)
        
        # Résultats avec métriques avancées
        for i, crypto in enumerate(sorted_results, 1):
            # Récupérer métriques avancées si disponibles
            adv_metrics = getattr(crypto, 'advanced_metrics', {})
            
            macd_signal = "+" if adv_metrics.get('macd', 0) > adv_metrics.get('macd_signal', 0) else "-"
            stoch_k = adv_metrics.get('stochastic_k', 50)
            patterns = ','.join(adv_metrics.get('patterns', []))[:14] or 'None'
            
            print(f"{i:<4} {crypto.symbol:<8} {crypto.name[:16]:<18} "
                  f"${crypto.price:<11.6f} {crypto.explosion_score:<5.1f} "
                  f"{crypto.volatility_7d:<6.1f}% {crypto.rsi:<4.1f} "
                  f"{macd_signal:<6} {stoch_k:<5.1f} {patterns:<15} "
                  f"{crypto.recommendation}")
        
        # Statistiques avancées
        print("\n📊 ANALYSE TECHNIQUE GLOBALE:")
        
        # Comptage des patterns
        all_patterns = []
        for crypto in sorted_results:
            adv_metrics = getattr(crypto, 'advanced_metrics', {})
            all_patterns.extend(adv_metrics.get('patterns', []))
            
        pattern_counts = {}
        for pattern in all_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
        print(f"   • Patterns détectés: {pattern_counts}")
        print(f"   • Moyenne score avancé: {sum(r.explosion_score for r in sorted_results) / len(sorted_results):.1f}/100")
        print(f"   • Cryptos en survente (RSI<30): {len([r for r in sorted_results if r.rsi < 30])}")
        print(f"   • Signaux MACD positifs: {len([r for r in sorted_results if getattr(r, 'advanced_metrics', {}).get('macd', 0) > getattr(r, 'advanced_metrics', {}).get('macd_signal', 0)])}")

    def export_advanced_results(self, results, filename=None):
        """Exporte les résultats avancés en CSV"""
        if not filename:
            filename = f"crypto_advanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        import pandas as pd
        
        data = []
        for crypto in results:
            adv_metrics = getattr(crypto, 'advanced_metrics', {})
            
            row = {
                'Symbol': crypto.symbol,
                'Name': crypto.name,
                'Price': crypto.price,
                'Enhanced_Score': crypto.explosion_score,
                'Volatility_7d': crypto.volatility_7d,
                'RSI': crypto.rsi,
                'MACD': adv_metrics.get('macd', 0),
                'MACD_Signal': adv_metrics.get('macd_signal', 0),
                'Stochastic_K': adv_metrics.get('stochastic_k', 50),
                'Bollinger_Position': adv_metrics.get('bollinger_position', 'normal'),
                'Sharpe_Ratio': adv_metrics.get('sharpe_ratio', 0),
                'Max_Drawdown': adv_metrics.get('max_drawdown', 0),
                'Patterns': ','.join(adv_metrics.get('patterns', [])),
                'Exchange_Diversity': adv_metrics.get('exchange_diversity', 50),
                'Recommendation': crypto.recommendation,
                'Market_Cap': crypto.market_cap,
                'Volume_24h': crypto.volume_24h
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"📁 Analyse avancée exportée dans {filename}")

    async def run_monitoring_mode(self, args):
        """Mode surveillance continue"""
        print("👁️  MODE SURVEILLANCE ACTIVÉ")
        print("=" * 50)
        print(f"Surveillance de {args.limit} cryptos toutes les {args.monitor_interval} minutes")
        print("Appuyez sur Ctrl+C pour arrêter\n")
        
        try:
            while True:
                print(f"\n🕐 {datetime.now().strftime('%H:%M:%S')} - Nouvelle analyse...")
                
                if args.advanced:
                    await self.run_advanced_analysis(args)
                else:
                    await self.run_basic_analysis(args)
                    
                print(f"⏳ Attente de {args.monitor_interval} minutes...")
                await asyncio.sleep(args.monitor_interval * 60)
                
        except KeyboardInterrupt:
            print("\n🛑 Surveillance arrêtée par l'utilisateur")

async def main():
    parser = argparse.ArgumentParser(
        description='🚀 Crypto Volatility Analyzer pour Raspberry Pi 5',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXEMPLES D'UTILISATION:

  Analyse standard:
    python3 crypto_launcher.py --limit 200 --top 30

  Analyse avancée (recommandé):
    python3 crypto_launcher.py --advanced --limit 100 --min-score 60

  Recherche de gems:
    python3 crypto_launcher.py --advanced --min-market-cap 1000000 --min-score 70

  Mode surveillance:
    python3 crypto_launcher.py --monitor --monitor-interval 30

  Export complet:
    python3 crypto_launcher.py --advanced --export --limit 500
        """
    )
    
    # Arguments principaux
    parser.add_argument('--limit', type=int, default=200,
                       help='Nombre de cryptos à analyser (défaut: 200)')
    parser.add_argument('--min-market-cap', type=int, default=1000000,
                       help='Market cap minimum en USD (défaut: 1M$)')
    parser.add_argument('--sort-by', 
                       choices=['explosion_score', 'volatility_7d', 'volatility_30d', 'price_change_7d'],
                       default='explosion_score',
                       help='Critère de tri (défaut: explosion_score)')
    parser.add_argument('--top', type=int, default=50,
                       help='Nombre de résultats à afficher (défaut: 50)')
    parser.add_argument('--min-score', type=float, default=0,
                       help='Score minimum pour filtrer (défaut: 0)')
    
    # Mode d'analyse
    parser.add_argument('--advanced', action='store_true',
                       help='🚀 Active l\'analyse technique avancée (RECOMMANDÉ)')
    parser.add_argument('--monitor', action='store_true',
                       help='👁️  Mode surveillance continue')
    parser.add_argument('--monitor-interval', type=int, default=30,
                       help='Interval de surveillance en minutes (défaut: 30)')
    
    # Export
    parser.add_argument('--export', action='store_true',
                       help='Exporter les résultats en CSV')
    parser.add_argument('--export-file', type=str,
                       help='Nom du fichier d\'export personnalisé')
    
    # Options de debug
    parser.add_argument('--debug', action='store_true',
                       help='Mode debug avec logs détaillés')
    
    args = parser.parse_args()
    
    # Validation des arguments
    if args.limit > 1000:
        print("⚠️  Attention: Analyser plus de 1000 cryptos peut prendre beaucoup de temps")
        
    if args.advanced and args.limit > 200:
        print("⚠️  Mode avancé: Limiter à 200 cryptos pour des performances optimales")
    
    # Affichage des paramètres
    print("🚀 CRYPTO VOLATILITY ANALYZER - RASPBERRY PI 5")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   • Mode: {'🔬 AVANCÉ' if args.advanced else '📈 STANDARD'}")
    print(f"   • Cryptos à analyser: {args.limit}")
    print(f"   • Market cap minimum: ${args.min_market_cap:,}")
    print(f"   • Score minimum: {args.min_score}")
    print(f"   • Top à afficher: {args.top}")
    print(f"   • Tri par: {args.sort_by}")
    if args.monitor:
        print(f"   • Surveillance: Oui (toutes les {args.monitor_interval}min)")
    print("")
    
    launcher = CryptoLauncher()
    
    try:
        start_time = time.time()
        
        if args.monitor:
            await launcher.run_monitoring_mode(args)
        elif args.advanced:
            await launcher.run_advanced_analysis(args)
        else:
            await launcher.run_basic_analysis(args)
            
        end_time = time.time()
        
        if not args.monitor:
            print(f"\n⏱️  Analyse terminée en {end_time - start_time:.2f} secondes")
            print("✨ Merci d'avoir utilisé Crypto Volatility Analyzer!")
            
    except KeyboardInterrupt:
        print("\n👋 Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur critique: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())