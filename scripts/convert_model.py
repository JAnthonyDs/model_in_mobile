import torch
import torch.nn as nn
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import numpy as np
import os
import argparse

from torchvision.models.segmentation import fcn_resnet50
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.transforms import v2 as T

def create_representative_dataloader(img_size, batch_size=1):
    """Cria um DataLoader para o dataset representativo."""
    input_transform = T.Compose([
        T.ToImage(),
        T.Resize(img_size),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = torchvision.datasets.OxfordIIITPet(
        root=".",
        split="trainval",
        # target_types="segmentation",
        download=True, # Baixará se não existir
        transform=input_transform,
    )
    
    subset_indices = torch.arange(200)
    subset_dataset = Subset(dataset, subset_indices)
    
    loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)
    return loader

def main(args):
    """
    Executa o pipeline completo de conversão do modelo:
    PyTorch -> ONNX -> TensorFlow -> TFLite INT8 Quantizado.
    """
    print("Iniciando o processo de conversão do modelo...")

    print("\n[PASSO 1/3] Convertendo PyTorch para ONNX...")
    model = fcn_resnet50()
    model.classifier[4] = nn.Conv2d(512, args.num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    model.load_state_dict(torch.load(args.input_model_path, map_location=torch.device('cpu')), strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    onnx_model_path = "pet_segmentation.onnx"
    torch.onnx.export(
        model, dummy_input, onnx_model_path,
        input_names=['input'], output_names=['output'], opset_version=11
    )
    print(f"Modelo ONNX salvo em: {onnx_model_path}")

    print("\n[PASSO 2/3] Convertendo ONNX para TensorFlow SavedModel...")
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_model_path = "pet_segmentation_tf"
    tf_rep.export_graph(tf_model_path)
    print(f"Modelo TensorFlow SavedModel salvo em: {tf_model_path}")

    print("\n[PASSO 3/3] Convertendo para TFLite INT8 com quantização...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    representative_loader = create_representative_dataloader(img_size=(args.img_size, args.img_size))
    
    def representative_dataset_gen():
        for image, _ in representative_loader:
            yield [image.numpy().astype(np.float32)]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    tflite_model_quant = converter.convert()

    with open(args.output_model_path, "wb") as f:
        f.write(tflite_model_quant)

    print("\nConversão para TFLite INT8 concluída com sucesso!")
    print(f"Modelo quantizado salvo em: {args.output_model_path}")

    onnx_size = os.path.getsize(onnx_model_path) / 1024 / 1024
    tflite_size = os.path.getsize(args.output_model_path) / 1024 / 1024
    print(f"\nTamanho do modelo ONNX: {onnx_size:.2f} MB")
    print(f"Tamanho do modelo TFLite INT8: {tflite_size:.2f} MB")
    print(f"Redução de tamanho: {100 * (1 - tflite_size / onnx_size):.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converte um modelo de segmentação PyTorch para TFLite INT8.")
    parser.add_argument(
        '--input-model-path', type=str, required=True,
        help="Caminho para o modelo PyTorch treinado (.pth)."
    )
    parser.add_argument(
        '--output-model-path', type=str, required=True,
        help="Caminho para salvar o modelo TFLite quantizado (.tflite)."
    )
    parser.add_argument('--num-classes', type=int, default=3, help="Número de classes da segmentação.")
    parser.add_argument('--img-size', type=int, default=128, help="Tamanho da imagem (altura e largura).")
    
    args = parser.parse_args()
    main(args)