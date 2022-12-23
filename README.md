# Sirius_CLIP

## Best prompts
1. scary house
2. house on hill | van gogh style
3. A pastoral landscape painting by Rembrandt
4. a home built in a huge Soap bubble, windows, doors, porches, awnings, middle of SPACE, cyberpunk lights, Hyper Detail, 8K, HD, Octane Rendering, Unreal Engine, V-Ray, full hd
5. dense woodland landscape, English forest, Irish forest, scottish forest, perspective, folklore,King Arthur, Lord of the Rings, Game of Thrones. ultra photoreal , photographic, concept art, cinematic lighting, cinematic composition, rule of thirds , mysterious, eerie, cinematic lighting, ultra-detailed, ultrarealistic, photorealism, 8k, octane render, Albert Bierstadt

## Best preferences

### Augmentations
```py
class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn=32, cut_pow=1., genius_moment=1):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.2),
            K.RandomSharpness(0.5, p=0.3),
            K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.1),
            K.RandomAffine(degrees=90, translate=0.3, p=0.2, padding_mode='border'),
            K.RandomPerspective(0.5, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.25),
            K.RandomGrayscale(p=0.2),
        )
        self.noise_fac = 0.1
 
 
    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        
        if self.noise_fac:
            facs = batch.new_empty([len(batch), 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch
```
> cutn=64 is also good
### Optimizer
```py
optimizer = torch.optim.Adam([z], lr=0.3)
```
> training with 100 epochs
