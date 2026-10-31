import time, sys, torch
from PIL import Image
from sentence_transformers import SentenceTransformer
torch.set_num_threads(8)
t=time.time(); m=SentenceTransformer("google/embeddinggemma-2", device="cpu", model_kwargs={"torch_dtype": torch.float32}); print("load_s", round(time.time()-t,1))
print("modules", [type(x).__name__ for x in m])
print("prompts", list((m.prompts or {}).keys())[:12])
imgs=[Image.open(p).convert("RGB") for p in sys.argv[1:]]
texts=["a doctor talking to a patient in a clinic","smart glasses on a table","an operating room","a city street at night"]
for attempt in ("pil","dict"):
    try:
        t=time.time()
        inp = imgs if attempt=="pil" else [{"image": im} for im in imgs]
        ie=m.encode(inp, normalize_embeddings=True); dt=(time.time()-t)/len(imgs)
        te=m.encode(texts, prompt_name=None, normalize_embeddings=True)
        print(attempt, "ok shape", ie.shape, "s_per_image", round(dt,2))
        sim=ie@te.T
        for p,row in zip(sys.argv[1:],sim): print(p.split('/')[-2:], [round(float(x),3) for x in row])
        break
    except Exception as e:
        print(attempt, "ERR", type(e).__name__, str(e)[:300])
