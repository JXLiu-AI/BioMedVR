import torch
import clip
from PIL import Image

def load_image(path, preprocess, device):
    image = Image.open(path).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # image_path_a = "/home/yinan2/bio/AttrVR-main/MedicalData/RETINA/cataract/_4_1244424.jpg"
    image_path_a = "/home/yinan2/bio/AttrVR-main/MedicalData/RETINA/cataract/_14_7805520.jpg"
    image_path_b = "/home/yinan2/bio/AttrVR-main/MedicalData/RETINA/glaucoma/_15_1523968.jpg"
    text_prompts = ["The image may reveal diffuse cloudiness and decreased red reflex due to lens changes.", "The image may reveal notching of the rim and peripapillary atrophy."]

    image_a = load_image(image_path_a, preprocess, device)
    image_b = load_image(image_path_b, preprocess, device)

    text_tokens = clip.tokenize(text_prompts).to(device)

    with torch.no_grad():
        image_features_a = model.encode_image(image_a).float()
        image_features_b = model.encode_image(image_b).float()
        text_features = model.encode_text(text_tokens).float()

    image_features_a /= image_features_a.norm(dim=-1, keepdim=True)
    image_features_b /= image_features_b.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    sims_a = (image_features_a @ text_features.t()).squeeze(0)
    sims_b = (image_features_b @ text_features.t()).squeeze(0)

    for idx, prompt in enumerate(text_prompts):
        print(f"{image_path_a} vs '{prompt}' similarity: {sims_a[idx].item():.4f}")
        print(f"{image_path_b} vs '{prompt}' similarity: {sims_b[idx].item():.4f}")

if __name__ == "__main__":
    main()
