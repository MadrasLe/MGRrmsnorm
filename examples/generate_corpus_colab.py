# =============================================================================
# 🧠 GERADOR DE CORPUS SINTÉTICO — MegaGemm + Qwen 3 4B FP16 (Colab T4)
# =============================================================================
#
# Versão NOTEBOOK — coloque no Colab e rode célula por célula.
#
# Setup (primeira célula do Colab):
#   !git clone https://github.com/MadrasLe/MGRrmsnorm.git
#   %cd MGRrmsnorm
#   !pip install triton huggingface_hub safetensors transformers
#   !pip install -e .
#
# ── T4 (sm_75, Turing) — 16GB VRAM ──
#
#   Qwen3-4B FP16 ≈ 8GB → cabe com folga (~8GB livres pra KV+ativações)
#   FP16 nativo nos Tensor Cores da T4 → ~3x mais rápido que INT8 dequant
#   Sem overhead de quantização = throughput máximo
#
# =============================================================================

import torch
import json
import os
from datetime import datetime
import random
from tqdm.auto import tqdm
import gc
import time


# ================== CONFIG — EDITE AQUI ==================

TOPICS_FILE = "topics.txt"              # 1 tema por linha
OUTPUT_FILE = "synthetic_corpus.jsonl"
PROGRESS_FILE = "corpus_progress.json"

LIMIT = 2000                             # Quantos documentos gerar
BATCH_SIZE = 256                         # Docs por batch (sobra VRAM pra KV cache)
MAX_NEW_TOKENS = 800                   # ~800-1200 palavras
CHECKPOINT_EVERY = 10                   # Salvar progresso a cada N batches

# Tamanho dos textos
WORD_COUNT_MIN = 800
WORD_COUNT_MAX = 1200
WORD_RANGE = f"{WORD_COUNT_MIN}-{WORD_COUNT_MAX} palavras"

# ── Modelo ──
# Qwen 3 4B FP16 puro — Tensor Cores nativos da T4, zero overhead
MODEL_NAME = "Qwen/Qwen3-4B"
QUANTIZE = None                         # FP16 puro, sem quantização
N_GPU_LAYERS = -1                       # -1 = tudo na GPU

# ── Geração ──
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.1


# ================== CARREGA ENGINE ==================
quant_label = QUANTIZE or 'fp16'
print(f"🚀 Carregando {MODEL_NAME} em {quant_label}...")
print(f"   GPU layers: {'todas' if N_GPU_LAYERS == -1 else N_GPU_LAYERS}")
start_load = time.time()

from megagemm.engine import InferenceEngine

engine = InferenceEngine(
    MODEL_NAME,
    quantize=QUANTIZE,
    num_blocks=2300,
    n_gpu_layers=N_GPU_LAYERS,
    max_batch_size=BATCH_SIZE,
    max_seq_len=MAX_NEW_TOKENS + 500,   # geração + margem pro prompt
    kv_offload=True,
    gpu_window=128,# offload blocos frios → CPU RAM
    # kv_alloc='auto',                  # default: aloca baseado no workload
    # dtype=torch.float16,              # default
)

load_time = time.time() - start_load
print(f"✅ Modelo carregado em {load_time:.1f}s!")
print(f"   {engine}")


# ================== FUNÇÕES DE PROGRESSO ==================
def load_topics(topics_file: str) -> list:
    """Carrega temas do arquivo."""
    if not os.path.exists(topics_file):
        raise ValueError(f"Arquivo {topics_file} não encontrado!")
    with open(topics_file, 'r', encoding='utf-8') as f:
        topics = [line.strip() for line in f if line.strip()]
    return topics


def load_progress() -> dict:
    """Carrega progresso."""
    default = {"processed_topics": [], "total_generated": 0}
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data.setdefault("processed_topics", [])
            data.setdefault("total_generated", 0)
            return data
        except:
            return default
    return default


def save_progress(progress: dict):
    """Salva progresso."""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)


# ================== 20 ESTILOS DE PROMPT ==================
def create_document_prompt(topic: str) -> str:
    """Prompts genéricos que funcionam para qualquer tema."""
    styles = [
        # === TEXTOS FORMAIS ===
        f"Escreva um texto completo e detalhado sobre: {topic}\n\n- {WORD_RANGE}\n- Cubra os aspectos mais relevantes\n- Use linguagem clara e acessível\n- Português brasileiro",

        f"Explique de forma completa e aprofundada: {topic}\n\nInclua definições, contexto, exemplos e implicações práticas. {WORD_RANGE} em português brasileiro.",

        f"Apresente uma visão geral abrangente sobre: {topic}\n\nAborde o que é, por que é importante, como funciona e quais são as principais considerações. {WORD_RANGE} em português.",

        f"Faça uma análise completa de: {topic}\n\nExplore diferentes ângulos, apresente informações relevantes e ofereça uma perspectiva equilibrada. {WORD_RANGE} em português brasileiro.",

        f"Crie um documento educativo sobre: {topic}\n\nEstruture o conteúdo de forma lógica, incluindo conceitos-chave e exemplos. {WORD_RANGE} em português.",

        f"Apresente um panorama informativo de: {topic}\n\nCubra os pontos principais, relevância e aspectos práticos do tema. {WORD_RANGE} em português brasileiro.",

        # === CONVERSAS E DIÁLOGOS ===
        f"Crie uma conversa natural entre duas pessoas discutindo sobre: {topic}\n\nFormato de diálogo com nomes (Ex: João: ... / Maria: ...)\n- Conversa informal e autêntica\n- Troca de opiniões e informações\n- {WORD_RANGE} em português brasileiro",

        f"Simule uma entrevista com um especialista sobre: {topic}\n\nFormato pergunta e resposta (P: ... / R: ...)\n- 8-12 perguntas interessantes\n- Respostas detalhadas e informativas\n- {WORD_RANGE} em português brasileiro",

        f"Crie uma seção de perguntas e respostas sobre: {topic}\n\nFormato FAQ com 10-15 perguntas comuns\n- Perguntas que pessoas realmente fariam\n- Respostas claras e úteis\n- {WORD_RANGE} em português brasileiro",

        f"Simule um debate amigável entre duas pessoas com visões diferentes sobre: {topic}\n\nFormato de diálogo (Pessoa A: ... / Pessoa B: ...)\n- Argumentos de ambos os lados\n- Tom respeitoso\n- {WORD_RANGE} em português brasileiro",

        # === REDES SOCIAIS ===
        f"Crie uma thread de Twitter/X explicando: {topic}\n\n- 15-20 tweets conectados\n- Cada tweet com no máximo 280 caracteres\n- Use numeração (1/, 2/, etc)\n- Linguagem direta e engajante\n- Português brasileiro",

        f"Escreva um post detalhado de fórum/Reddit sobre: {topic}\n\n- Título atraente\n- Corpo do post com informações completas\n- Tom de comunidade online\n- Inclua TL;DR no final\n- {WORD_RANGE} em português brasileiro",

        f"Simule uma postagem viral sobre {topic} seguida de vários comentários de diferentes usuários.\n\n- 1 post principal\n- 10-15 comentários variados (concordando, discordando, perguntando, zoando)\n- Formato: @usuario: comentário\n- Português brasileiro informal",

        f"Crie o texto para um carrossel de Instagram/Stories sobre: {topic}\n\n- 8-10 slides\n- Cada slide com título + texto curto\n- Linguagem visual e direta\n- Call-to-action no final\n- Português brasileiro",

        f"Escreva um post profissional de LinkedIn sobre: {topic}\n\n- Gancho forte no início\n- Storytelling ou insights valiosos\n- Formatação com quebras de linha\n- Call-to-action ou pergunta no final\n- 800-1500 caracteres em português brasileiro",

        # === FORMATOS CRIATIVOS ===
        f"Crie um tutorial conversacional sobre: {topic}\n\n- Como se estivesse ensinando um amigo\n- Passos claros e numerados\n- Dicas e avisos importantes\n- {WORD_RANGE} em português brasileiro",

        f"Escreva uma review/opinião detalhada sobre: {topic}\n\n- Prós e contras\n- Experiência pessoal (simulada)\n- Recomendações finais\n- Tom autêntico e pessoal\n- {WORD_RANGE} em português brasileiro",

        f"Crie uma newsletter sobre: {topic}\n\n- Assunto atraente\n- Saudação pessoal\n- Conteúdo principal informativo\n- Seções com subtítulos\n- Despedida e CTA\n- {WORD_RANGE} em português brasileiro",

        f"Simule a transcrição de um episódio de podcast sobre: {topic}\n\n- Apresentador(a) explicando o tema\n- Tom conversacional e descontraído\n- Tangentes interessantes\n- {WORD_RANGE} em português brasileiro",

        f"Simule uma conversa de grupo de WhatsApp/Telegram discutindo: {topic}\n\n- 4-5 participantes com nomes diferentes\n- Mensagens curtas e naturais\n- Emojis ocasionais\n- Discussão orgânica com perguntas e respostas\n- Português brasileiro informal",
    ]
    return random.choice(styles)


# ================== GERAÇÃO COM MEGAGEMM ==================
def generate_document(topic: str) -> tuple:
    """
    Gera 1 documento para 1 tema via MegaGemm.

    MegaGemm aplica chat template automaticamente!
    Retorna: (doc_dict ou None, num_tokens)
    """
    prompt_text = create_document_prompt(topic)

    t0 = time.time()
    content = engine.generate(
        prompt_text,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
    )
    gen_time = time.time() - t0

    # Estima tokens pelo comprimento (MegaGemm retorna string direto)
    est_tokens = len(content.split()) * 1.3  # ~1.3 tokens/palavra pt-br

    if len(content) < 300:
        return None, int(est_tokens)

    doc = {
        "text": content,
        "topic": topic,
        "word_count": len(content.split()),
        "char_count": len(content),
        "est_tokens": int(est_tokens),
        "gen_time_s": round(gen_time, 2),
        "model": MODEL_NAME.split("/")[-1],
        "quantization": QUANTIZE,
        "engine": "megagemm",
        "timestamp": datetime.now().isoformat(),
    }
    return doc, int(est_tokens)


def generate_documents_batch(topics: list) -> tuple:
    """
    Gera documentos em batch via MegaGemm continuous batching.
    Retorna: (list[doc ou None], total_est_tokens)
    """
    prompts = [create_document_prompt(t) for t in topics]

    t0 = time.time()
    texts = engine.generate_batch(
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )
    gen_time = time.time() - t0

    results = []
    total_tokens = 0

    for i, content in enumerate(texts):
        est_tokens = int(len(content.split()) * 1.3)
        total_tokens += est_tokens

        if len(content) < 300:
            results.append(None)
            continue

        doc = {
            "text": content,
            "topic": topics[i],
            "word_count": len(content.split()),
            "char_count": len(content),
            "est_tokens": est_tokens,
            "gen_time_s": round(gen_time / len(topics), 2),
            "model": MODEL_NAME.split("/")[-1],
            "quantization": QUANTIZE,
            "engine": "megagemm",
            "timestamp": datetime.now().isoformat(),
        }
        results.append(doc)

    return results, total_tokens


# ================== EXECUÇÃO PRINCIPAL ==================
print("\n" + "=" * 60)
print("🧠 GERADOR DE CORPUS — MegaGemm Engine")
print("=" * 60)

# Carrega temas
all_topics = load_topics(TOPICS_FILE)
print(f"📋 Temas disponíveis: {len(all_topics)}")

# Carrega progresso
progress = load_progress()
processed_set = set(progress["processed_topics"])
print(f"📈 Progresso anterior: {progress['total_generated']} documentos")

# Filtra temas pendentes
pending_topics = [t for t in all_topics if t not in processed_set]

if not pending_topics:
    print("\n✅ Todos os temas já foram processados!")
else:
    random.shuffle(pending_topics)
    topics_to_process = pending_topics[:LIMIT]

    print(f"🔥 Gerando {len(topics_to_process)} documentos...")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🔧 Engine: MegaGemm | Quant: {QUANTIZE} | Model: {MODEL_NAME}\n")

    generated = 0
    failed = 0
    total_tokens = 0

    pbar = tqdm(total=len(topics_to_process), desc="Gerando Corpus")
    start_gen = time.time()

    for batch_start in range(0, len(topics_to_process), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(topics_to_process))
        batch_topics = topics_to_process[batch_start:batch_end]

        try:
            if len(batch_topics) == 1:
                # Single doc — usa generate() direto
                doc, batch_tokens = generate_document(batch_topics[0])
                total_tokens += batch_tokens

                if doc:
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    progress["processed_topics"].append(batch_topics[0])
                    progress["total_generated"] += 1
                    generated += 1
                else:
                    failed += 1
                pbar.update(1)
            else:
                # Streaming batch — progress bar atualiza em tempo real!
                prompts = [create_document_prompt(t) for t in batch_topics]
                t0 = time.time()

                for idx, content in engine.generate_batch_stream(
                    prompts,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    top_p=TOP_P,
                ):
                    topic = batch_topics[idx]
                    est_tokens = int(len(content.split()) * 1.3)
                    total_tokens += est_tokens

                    if len(content) >= 300:
                        doc = {
                            "text": content,
                            "topic": topic,
                            "word_count": len(content.split()),
                            "char_count": len(content),
                            "est_tokens": est_tokens,
                            "gen_time_s": round(time.time() - t0, 2),
                            "model": MODEL_NAME.split("/")[-1],
                            "quantization": QUANTIZE,
                            "engine": "megagemm",
                            "timestamp": datetime.now().isoformat(),
                        }
                        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                        progress["processed_topics"].append(topic)
                        progress["total_generated"] += 1
                        generated += 1
                    else:
                        failed += 1

                    pbar.update(1)

            elapsed = time.time() - start_gen
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                "gen": generated,
                "tok/s": f"{tok_per_sec:.1f}",
                "docs/min": f"{generated / elapsed * 60:.1f}" if elapsed > 0 else "0.0",
            })

        except Exception as e:
            print(f"\n⚠️ Erro no batch: {e}")
            # Fallback: tenta 1 por 1
            for topic in batch_topics:
                try:
                    doc, batch_tokens = generate_document(topic)
                    total_tokens += batch_tokens

                    if doc:
                        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                        progress["processed_topics"].append(topic)
                        progress["total_generated"] += 1
                        generated += 1
                    else:
                        failed += 1
                except Exception as e2:
                    print(f"   ❌ Falha individual [{topic[:40]}]: {e2}")
                    failed += 1

                pbar.update(1)

        # Checkpoint periódico
        batch_num = batch_start // BATCH_SIZE + 1
        if batch_num % CHECKPOINT_EVERY == 0:
            save_progress(progress)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()
    save_progress(progress)

    # ── Relatório Final ──
    total_time = time.time() - start_gen

    print("\n" + "=" * 60)
    print("📊 RELATÓRIO FINAL")
    print("=" * 60)
    print(f"✅ Gerados:     {generated}")
    print(f"❌ Falhados:    {failed}")
    print(f"📁 Arquivo:     {OUTPUT_FILE}")
    print(f"⚡ Tempo:       {total_time / 60:.1f} min")
    print(f"📈 Docs/min:    {generated / total_time * 60:.1f}" if total_time > 0 else "")
    print(f"⚡ Tokens/s:    {total_tokens / total_time:.1f}" if total_time > 0 else "")
    print(f"📊 Total tokens: {total_tokens:,}")
    print(f"🧠 Engine:       MegaGemm | {QUANTIZE.upper()}")
    print(f"\n📈 Total acumulado: {progress['total_generated']} documentos")

    remaining = len(pending_topics) - len(topics_to_process)
    if remaining > 0:
        print(f"\n👉 Temas restantes: {remaining}")
        print(f"   Mude LIMIT = {min(remaining, LIMIT)} e rode novamente")
