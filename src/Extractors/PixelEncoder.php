<?php

namespace Rubix\ML\Extractors;

use Intervention\Image\Image;
use Intervention\Image\ImageManager as Intervention;
use InvalidArgumentException;
use RuntimeException;

class PixelEncoder implements Extractor
{
    /**
     * The image will be scaled and cropped according to the setting of this
     * parameter which will have an effect on the size of the outpput vector.
     *
     * @var array
     */
    protected $size;

    /**
     * The number of channels to encode. Each channel requires width x height
     * number of features.
     *
     * @var bool
     */
    protected $channels;

    /**
     * The Intervention image manager instance.
     *
     * @var \Intervention\Image\ImageManager
     */
    protected $intervention;

    /**
     * @param  array  $size
     * @param  bool  $rgb
     * @param  string  $driver
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $size = [32, 32], bool $rgb = true,
                                int $sharpen = 0, string $driver = 'gd')
    {
        if (count($size) !== 2) {
            throw new InvalidArgumentException('Size must have a width and a'
                . ' height.');
        }

        foreach ($size as $dimension) {
            if ($dimension < 1) {
                throw new InvalidArgumentException('Width and height must be'
                    . ' greater than 1 pixel.');
            }
        }

        if ($sharpen < 0 or $sharpen > 100) {
            throw new InvalidArgumentException('Sharpness factor must be'
                . ' between 0 and 100');
        }

        $this->size = $size;
        $this->channels = $rgb ? 3 : 1;
        $this->sharpen = $sharpen;
        $this->intervention = new Intervention(['driver' => $driver]);
    }

    /**
     * @param  array  $samples
     * @return void
     */
    public function fit(array $samples) : void
    {
        //
    }

    /**
     * Extract the pixel data from each sample image and represent it as a 1-d
     * vector of size width x height.
     *
     * @param  array  $samples
     * @throws \RuntimeException
     * @return array
     */
    public function extract(array $samples) : array
    {
        $vectors = [];

        foreach ($samples as $sample) {
            if (is_resource($sample)) {
                $image = $this->intervention->make($sample);

                if ($this->channels === 1) {
                    $image = $image->greyscale();
                }

                $image->fit(...$this->size)
                    ->sharpen($this->sharpen);

                $vectors[] = $this->vectorize($image);
            }
        }

        return $vectors;
    }

    /**
     * Convert a image into a vector of color channel data.
     *
     * @param  \Intervention\Image\Image  $image
     * @return array
     */
    public function vectorize(Image $image) : array
    {
        $image = $image->getCore();

        $vector = [];

        for ($x = 0; $x < $this->size[0]; $x++) {
            for ($y = 0; $y < $this->size[1]; $y++) {
                $rgba = imagecolorsforindex($image,
                    imagecolorat($image, $x, $y));

                $vector = array_merge($vector,
                    array_values(array_slice($rgba, 0, $this->channels)));
            }
        }

        return $vector;
    }
}
