#!/bin/bash

echo "🚀 Configuration du Crypto Volatility Analyzer pour Raspberry Pi 5"
echo "=================================================================="

# Vérification que nous sommes sur un système compatible
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé"
    exit 1
fi

# Mise à jour du système
echo "📦 Mise à jour du système..."
sudo apt update && sudo apt upgrade -y

# Installation des dépendances système
echo "🔧 Installation des dépendances système..."
sudo apt install -y python3-pip python3-venv git curl

# Création de l'environnement virtuel
echo "🐍 Création de l'environnement virtuel Python..."
python3 -m venv crypto_env
source crypto_env/bin/activate

# Installation des packages Python
echo "📚 Installation des packages Python..."
pip install --upgrade pip
pip install -r requirements.txt

# Création du script de lancement principal
echo "📝 Création du script de lancement..."
cat > run_crypto_analyzer.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source crypto_env/bin/activate
python3 crypto_launcher.py "$@"
EOF

# Création d'un script pour l'analyse avancée
cat > run_advanced_analysis.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source crypto_env/bin/activate
python3 crypto_launcher.py --advanced "$@"
EOF

# Création d'un script de surveillance
cat > run_monitoring.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source crypto_env/bin/activate
python3 crypto_launcher.py --monitor --advanced --monitor-interval 15 "$@"
EOF

chmod +x run_crypto_analyzer.sh
chmod +x run_advanced_analysis.sh 
chmod +x run_monitoring.sh

# Création d'aliases dans .bashrc
echo "⚙️  Configuration des aliases..."
if ! grep -q "crypto-analyzer" ~/.bashrc; then
    echo "alias crypto-analyzer='$(pwd)/run_crypto_analyzer.sh'" >> ~/.bashrc
    echo "alias crypto-advanced='$(pwd)/run_advanced_analysis.sh'" >> ~/.bashrc
    echo "alias crypto-monitor='$(pwd)/run_monitoring.sh'" >> ~/.bashrc
    echo "✅ Aliases ajoutés à .bashrc"
fi

# Création d'un menu interactif
cat > crypto_menu.sh << 'EOF'
#!/bin/bash
echo "🚀 CRYPTO VOLATILITY ANALYZER - MENU PRINCIPAL"
echo "=============================================="
echo ""
echo "1) 📈 Analyse Standard (rapide)"
echo "2) 🔬 Analyse Avancée (recommandée)" 
echo "3) 💎 Recherche de Gems (petites caps)"
echo "4) 👁️  Mode Surveillance"
echo "5) 📊 Export complet"
echo "6) ❓ Aide"
echo "0) Quitter"
echo ""
read -p "Choisissez une option (0-6): " choice

case $choice in
    1)
        echo "🏃 Lancement analyse standard..."
        ./run_crypto_analyzer.sh --limit 200 --top 30
        ;;
    2)
        echo "🔬 Lancement analyse avancée..."
        ./run_advanced_analysis.sh --limit 100 --top 25 --min-score 50
        ;;
    3)
        echo "💎 Recherche de gems..."
        ./run_advanced_analysis.sh --min-market-cap 1000000 --limit 500 --min-score 70 --top 15
        ;;
    4)
        echo "👁️  Démarrage surveillance..."
        ./run_monitoring.sh
        ;;
    5)
        echo "📊 Export complet..."
        ./run_advanced_analysis.sh --limit 300 --export --min-score 30
        ;;
    6)
        echo "❓ AIDE - EXEMPLES DE COMMANDES:"
        echo ""
        echo "Analyse de base:"
        echo "  crypto-analyzer --limit 100 --top 20"
        echo ""
        echo "Analyse avancée:"
        echo "  crypto-advanced --min-score 60 --top 15"
        echo ""
        echo "Recherche gems:"
        echo "  crypto-advanced --min-market-cap 5000000 --min-score 75"
        echo ""
        echo "Surveillance:"
        echo "  crypto-monitor"
        echo ""
        ;;
    0)
        echo "👋 Au revoir!"
        exit 0
        ;;
    *)
        echo "❌ Option invalide"
        ;;
esac
EOF

chmod +x crypto_menu.sh

echo ""
echo "🎉 Installation terminée avec succès !"
echo ""
echo "📋 MÉTHODES D'UTILISATION:"
echo ""
echo "1️⃣  MENU INTERACTIF (recommandé pour débuter):"
echo "   ./crypto_menu.sh"
echo ""
echo "2️⃣  LIGNE DE COMMANDE (après source ~/.bashrc):"
echo "   • crypto-analyzer          # Analyse standard"
echo "   • crypto-advanced          # Analyse avancée"  
echo "   • crypto-monitor           # Surveillance continue"
echo ""
echo "3️⃣  SCRIPTS DIRECTS:"
echo "   • ./run_crypto_analyzer.sh    # Script principal"
echo "   • ./run_advanced_analysis.sh  # Analyse avancée"
echo "   • ./run_monitoring.sh         # Surveillance"
echo ""
echo "🔧 EXEMPLES DE COMMANDES AVANCÉES:"
echo "   crypto-advanced --limit 150 --min-score 65 --top 20"
echo "   crypto-analyzer --min-market-cap 10000000 --export"
echo "   crypto-monitor --monitor-interval 20"
echo ""
echo "📁 FICHIERS CRÉÉS:"
echo "   • crypto_analyzer.py       # Analyseur de base"
echo "   • advanced_analyzer.py     # Module avancé"
echo "   • crypto_launcher.py       # Interface principale"
echo "   • crypto_menu.sh           # Menu interactif"
echo "   • *.csv                    # Exports (après utilisation)"
echo "   • crypto_analyzer.log      # Logs"
echo ""
echo "🚀 Pour commencer: source ~/.bashrc && ./crypto_menu.sh"
echo ""