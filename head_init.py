"""
head_init.py — Final layer initialization (student-implemented).

Students: Implement `init_last_layer` to control how the new classification
head is initialized before fine-tuning begins. The skeleton below uses
Kaiming uniform weights and zero bias — you are expected to experiment with
alternatives (e.g. Xavier, orthogonal, small-scale random, learned bias init).
"""
import torch
import torch.nn as nn
import torchvision.models as models

# Mapping from CIFAR100 to ImageNet labels
cifar100_to_imagenet = [
    ["Granny Smith", "custard apple"],
    ["goldfish", "anemone fish", "rock beauty"],
    ["cradle", "crib", "bassinet"],
    ["brown bear", "American black bear", "ice bear", "sloth bear"],
    ["beaver"],
    ["four-poster", "crib", "bassinet"],
    ["bee", "apiary", "honeycomb"],
    ["tiger beetle", "ladybug", "ground beetle", "long-horned beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil"],
    ["mountain bike", "bicycle-built-for-two"],
    ["water bottle", "pop bottle", "beer bottle", "pill bottle", "wine bottle"],
    ["mixing bowl", "soup bowl"],
    ["ballplayer", "groom"],
    ["suspension bridge", "steel arch bridge", "viaduct"],
    ["school bus", "minibus", "trolleybus"],
    ["monarch", "ringlet", "cabbage butterfly", "sulphur butterfly", "lycaenid"],
    ["Arabian camel"],
    ["milk can", "can opener"],
    ["castle", "palace"],
    ["walking stick", "centipede", "grasshopper"],
    ["ox", "water buffalo", "bison"],
    ["folding chair", "rocking chair", "barber chair", "throne"],
    ["chimpanzee"],
    ["analog clock", "digital clock", "wall clock"],
    ["bubble", "geyser"],
    ["cockroach"],
    ["studio couch"],
    ["Dungeness crab", "rock crab", "fiddler crab", "king crab", "hermit crab"],
    ["African crocodile", "American alligator"],
    ["cup", "coffee mug", "measuring cup", "goblet"],
    ["triceratops"],
    ["killer whale", "dugong", "sea lion"],
    ["African elephant", "Indian elephant", "tusker"],
    ["tench", "coho", "sturgeon", "gar"],
    ["valley", "lakeside", "promontory"],
    ["red fox", "kit fox", "Arctic fox", "grey fox"],
    ["gown", "maillot"],
    ["hamster"],
    ["mobile home", "boathouse", "birdhouse", "church", "monastery"],
    ["wallaby"],
    ["computer keyboard", "typewriter keyboard"],
    ["table lamp", "lampshade"],
    ["lawn mower"],
    ["leopard", "snow leopard"],
    ["lion"],
    ["banded gecko", "common iguana", "American chameleon", "whiptail", "agama", "frilled lizard", "alligator lizard", "Gila monster", "green lizard", "African chameleon"],
    ["American lobster", "spiny lobster"],
    ["groom", "ballplayer"],
    ["buckeye", "acorn"],
    ["motor scooter", "moped"],
    ["alp", "cliff", "promontory", "volcano", "valley"],
    ["mouse"],
    ["mushroom", "coral fungus", "agaric", "gyromitra", "stinkhorn", "earthstar", "hen-of-the-woods", "bolete"],
    ["acorn"],
    ["orange"],
    ["yellow lady's slipper"],
    ["otter"],
    ["banana", "pineapple"],
    ["Granny Smith", "custard apple", "fig"],
    ["pickup"],
    ["acorn", "buckeye", "lumbermill"],
    ["airliner", "warplane", "wing", "plane"],
    ["plate", "plate rack"],
    ["rapeseed", "daisy"],
    ["porcupine"],
    ["mink", "weasel", "armadillo"],
    ["wood rabbit", "hare", "Angora"],
    ["skunk", "badger", "polecat"],
    ["electric ray", "stingray"],
    ["street sign", "traffic light", "parking meter"],
    ["missile", "projectile", "space shuttle"],
    ["hip", "vase"],
    ["seashore", "sandbar", "lakeside", "coral reef"],
    ["sea lion"],
    ["great white shark", "tiger shark", "hammerhead"],
    ["mouse", "weasel", "mink"],
    ["skunk"],
    ["obelisk", "dome"],
    ["snail"],
    ["thunder snake", "ringneck snake", "hognose snake", "green snake", "king snake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "rock python", "Indian cobra", "green mamba", "sea snake", "horned viper", "diamondback", "sidewinder"],
    ["black and gold garden spider", "barn spider", "garden spider", "black widow", "tarantula", "wolf spider", "spider web"],
    ["fox squirrel"],
    ["streetcar", "trolleybus"],
    ["daisy", "rapeseed"],
    ["bell pepper"],
    ["dining table", "desk", "pool table"],
    ["tank", "half track"],
    ["cellular telephone", "dial telephone", "pay-phone"],
    ["television", "monitor", "screen", "home theater"],
    ["tiger"],
    ["tractor"],
    ["steam locomotive", "electric locomotive", "bullet train", "passenger car"],
    ["tench", "coho", "sturgeon", "gar"],
    ["vase", "daisy", "yellow lady's slipper"],
    ["loggerhead", "leatherback turtle", "mud turtle", "terrapin", "box turtle"],
    ["wardrobe", "chiffonier"],
    ["grey whale", "killer whale"],
    ["lakeside", "valley"],
    ["timber wolf", "white wolf", "red wolf", "coyote"],
    ["gown", "maillot", "kimono"],
    ["flatworm", "nematode"],
]

def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the weights and bias of the final classification layer in-place.

    This function is called once during model construction (see model.py).
    Modify it to experiment with different initialization strategies and observe
    their effect on the "initialized head" evaluation checkpoint.

    Args:
        layer: The ``nn.Linear`` layer that serves as the new CIFAR100 head.
               Modifies the layer in-place; return value is ignored.

    Student task:
        Replace or extend the skeleton below. Some strategies to consider:
          - ``nn.init.xavier_uniform_``  — preserves variance across layers
          - ``nn.init.orthogonal_``      — encourages diverse feature directions
          - Small-scale init (e.g. scale weights by 0.01) — conservative start
          - Non-zero bias init           — useful when class priors are known
    """
    # -------------------------------------------------------------------------
    # STUDENT: Replace or extend the initialization below.
    # -------------------------------------------------------------------------

    with torch.no_grad():
        torch.manual_seed(42)
        
        pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        fc_weights = pretrained.fc.weight.detach()
        fc_bias = pretrained.fc.bias.detach()
        
        name_to_index = {cat_name: idx for idx, cat_name in enumerate(models.ResNet18_Weights.IMAGENET1K_V1.meta['categories'])}
        
        aggregated_weights = []
        aggregated_bias = []
        
        for source_cats in cifar100_to_imagenet:
            source_indices = [name_to_index[cat] for cat in source_cats]
            
            weight_mean = fc_weights[source_indices].mean(dim=0)
            weight_noise = torch.empty_like(weight_mean).uniform_(-0.02, 0.02)
            
            aggregated_weights.append(weight_mean + weight_noise)
            aggregated_bias.append(fc_bias[source_indices].mean())
        
        layer.weight.copy_(torch.stack(aggregated_weights).to(layer.weight.dtype))
        layer.bias.copy_(torch.stack(aggregated_bias).to(layer.bias.dtype))


    # -------------------------------------------------------------------------