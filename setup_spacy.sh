#!/bin/bash

# Script para configurar o spaCy corretamente para o projeto PixelBot
# Este script resolve problemas de compatibilidade entre versões do spaCy e seus modelos

echo "🔧 Configurando spaCy para o projeto PixelBot..."

# Ativar ambiente virtual se existir
if [ -d "env" ]; then
    echo "📦 Ativando ambiente virtual..."
    source env/bin/activate
else
    echo "❌ Ambiente virtual não encontrado. Execute 'python -m venv env' primeiro."
    exit 1
fi

# Verificar se o spaCy está instalado
if ! python -c "import spacy" 2>/dev/null; then
    echo "📥 Instalando spaCy..."
    pip install spacy==3.7.2
else
    echo "✅ spaCy já está instalado"
fi

# Verificar versão do spaCy
SPACY_VERSION=$(python -c "import spacy; print(spacy.__version__)" 2>/dev/null)
echo "📋 Versão do spaCy: $SPACY_VERSION"

# Desinstalar modelo antigo se existir
echo "🗑️  Removendo modelo spaCy antigo (se existir)..."
pip uninstall en-core-web-sm -y 2>/dev/null

# Instalar modelo compatível
echo "📥 Instalando modelo spaCy en_core_web_sm..."
python -m spacy download en_core_web_sm

# Testar se está funcionando
echo "🧪 Testando instalação..."
if python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✅ spaCy model loaded successfully')" 2>/dev/null; then
    echo "🎉 spaCy configurado com sucesso!"
    echo "✅ O script load_sources.py agora deve funcionar corretamente"
else
    echo "❌ Erro ao carregar modelo spaCy"
    exit 1
fi

echo ""
echo "🚀 Para testar, execute:"
echo "   python load_sources.py --file_path data/ --table_key document"
