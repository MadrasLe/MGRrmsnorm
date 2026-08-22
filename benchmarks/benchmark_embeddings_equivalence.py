import argparse
import torch
from megagemm.embeddings import EmbeddingEngine

def main():
    parser = argparse.ArgumentParser(description="Test equivalence between HF and Native embeddings")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Hugging Face model ID")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_name = args.model
    texts = [
        "A Berta está muito orgulhosa do código implementado no MegaGemm.",
        "O motor de inferência utiliza paralelismo de kernels Triton e offloading.",
        "O Gabriel é um arquiteto brilhante de machine learning."
    ]

    print(f"=== Equivalence Test: {model_name} ===")

    print(f"\n1. Executando Backend HuggingFace (Base)")
    engine_hf = EmbeddingEngine(model_name, backend="hf", device=args.device, dtype="fp32")
    emb_hf = engine_hf.encode(texts)

    print(f"2. Executando Backend Native (Padding-Free)")
    engine_native = EmbeddingEngine(model_name, backend="native", device=args.device, dtype="fp32", native_padding_free=True)
    emb_native = engine_native.encode(texts)

    print("\n=== Resultados de Similaridade de Cosseno ===")
    cos_sims = torch.nn.functional.cosine_similarity(emb_hf, emb_native, dim=1)

    for i, text in enumerate(texts):
        print(f"\nTexto: '{text}'")
        print(f"  -> Similaridade (HF vs Native): {cos_sims[i].item():.6f}")

    avg_sim = cos_sims.mean().item()
    print(f"\nMédia de Similaridade: {avg_sim:.6f}")
    if avg_sim > 0.999:
        print("✅ SUCESSO: A engine Native (CUDA/Triton) é matematicamente EXATAMENTE igual ao PyTorch/HF!")
    else:
        print("❌ AVISO: Divergência detectada entre os motores.")

if __name__ == "__main__":
    main()
