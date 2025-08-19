# Load Sources - Carregamento de Arquivos para o Monitor Virtual

Este script permite carregar arquivos de uma pasta para as tabelas do Pixeltable, similar ao endpoint de upload da aplicação web, mas via linha de comando.

## Funcionalidades

- **Detecção automática de tipos de arquivo**: O script pode detectar automaticamente o tipo de arquivo baseado na extensão
- **Carregamento por tipo específico**: Você pode especificar um tipo específico de arquivo para carregar
- **Processamento recursivo**: Processa arquivos em subdiretórios
- **Logging detalhado**: Fornece logs informativos sobre o processo

## Tipos de Arquivo Suportados

- **Documentos**: PDF, TXT, MD, HTML, XML
- **Imagens**: JPG, JPEG, PNG
- **Vídeos**: MP4, MOV, AVI
- **Áudios**: MP3, WAV, M4A
- **Tabelas**: CSV, XLSX

## Uso

### 1. Detecção Automática (Recomendado)

Para carregar todos os tipos de arquivo de uma pasta, detectando automaticamente o tipo:

```bash
python load_sources.py --file_path /caminho/para/sua/pasta
```

### 2. Tipo Específico

Para carregar apenas um tipo específico de arquivo:

```bash
# Apenas documentos
python load_sources.py --file_path /caminho/para/sua/pasta --table_key document

# Apenas imagens
python load_sources.py --file_path /caminho/para/sua/pasta --table_key image

# Apenas vídeos
python load_sources.py --file_path /caminho/para/sua/pasta --table_key video

# Apenas áudios
python load_sources.py --file_path /caminho/para/sua/pasta --table_key audio

# Apenas tabelas
python load_sources.py --file_path /caminho/para/sua/pasta --table_key tabular
```

### 3. Com User ID Personalizado

```bash
python load_sources.py --file_path /caminho/para/sua/pasta --user_id meu_usuario_123
```

## Exemplos Práticos

### Carregar todos os arquivos da pasta data/

```bash
python load_sources.py --file_path data/
```

### Carregar apenas PDFs de uma pasta específica

```bash
python load_sources.py --file_path /home/usuario/documentos/ --table_key document
```

### Carregar imagens com user_id específico

```bash
python load_sources.py --file_path /home/usuario/fotos/ --table_key image --user_id usuario_456
```

## Estrutura das Tabelas

O script insere os arquivos nas seguintes tabelas do Pixeltable:

- **Documentos**: `agents.collection`
- **Imagens**: `agents.images`
- **Vídeos**: `agents.videos`
- **Áudios**: `agents.audios`
- **Tabelas**: `agents.tabular`

Cada registro contém:

- Caminho do arquivo
- UUID único
- Timestamp de inserção
- ID do usuário

## Logs

O script fornece logs detalhados incluindo:

- Número de arquivos encontrados
- Arquivos processados com sucesso
- Erros encontrados
- Resumo final do processamento

## Diferenças do Endpoint Web

- **Linha de comando**: Execução via terminal
- **Pasta inteira**: Processa todos os arquivos de uma pasta
- **Detecção automática**: Pode detectar tipos automaticamente
- **Sem interface web**: Não requer servidor web rodando

## Solução de Problemas

### Erro spaCy Model

Se você encontrar o erro `Failed to load spaCy model: en_core_web_sm`, execute o script de configuração:

```bash
./setup_spacy.sh
```

Este script irá:

1. Instalar a versão correta do spaCy (3.7.2)
2. Baixar o modelo de linguagem compatível
3. Testar se tudo está funcionando

### Alternativa Manual

Se preferir fazer manualmente:

```bash
# Ativar ambiente virtual
source env/bin/activate

# Instalar versão correta do spaCy
pip install spacy==3.7.2

# Baixar modelo de linguagem
python -m spacy download en_core_web_sm

# Testar
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('OK')"
```
