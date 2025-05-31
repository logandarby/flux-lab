export interface TextureSwapBuffer {
  front: GPUTexture;
  back: GPUTexture;
}

export interface TextureBuffer {
  front: GPUTexture;
}

export class TextureManager<TextureID extends string | number> {
  private readonly textures = new Map<
    TextureID,
    TextureSwapBuffer | TextureBuffer
  >();

  constructor(private readonly device: GPUDevice) {}

  public createTexture(
    name: TextureID,
    descriptor: GPUTextureDescriptor
  ): void {
    if (this.textures.has(name)) {
      throw new Error(`Texture ${name} already exists inside TextureManager!`);
    }
    this.textures.set(name, {
      front: this.device.createTexture(descriptor),
    });
  }

  public createPingPongTexture(
    name: TextureID,
    descriptor: GPUTextureDescriptor
  ): void {
    if (this.textures.has(name)) {
      throw new Error(
        `Texture "${name}" already exists inside TextureManager!`
      );
    }
    const texture1 = this.device.createTexture(descriptor);
    const texture2 = this.device.createTexture(descriptor);
    this.textures.set(name, {
      front: texture1,
      back: texture2,
    });
  }

  public swap(name: TextureID): void {
    const textures = this.textures.get(name);
    if (!textures) {
      throw new Error(`Texture '${name}' not found`);
    }
    if (!("back" in textures)) {
      throw new Error(`Cannot swap static single texture ${name}`);
    }
    const { front, back } = textures;
    this.textures.set(name, {
      front: back,
      back: front,
    });
  }

  public getCurrentTexture(name: TextureID): GPUTexture {
    const frontTexture = this.textures.get(name);
    if (!frontTexture) {
      throw new Error(`Texture "${name}" does not exist`);
    }
    return frontTexture.front;
  }

  public getBackTexture(name: TextureID): GPUTexture {
    const backTexture = this.textures.get(name);
    if (!backTexture) {
      throw new Error(`Texture ${name} does not exist`);
    }
    if (!("back" in backTexture)) {
      throw new Error(
        `Texture ${name} is not a ping-pong texture-- it has no back buffer, only a front`
      );
    }
    return backTexture.back;
  }
}
