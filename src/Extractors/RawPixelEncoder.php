<?php

namespace Rubix\ML\Extractors;

use Intervention\Image\Image;
use Intervention\Image\ImageManager;
use InvalidArgumentException;
use RuntimeException;

/**
 * Raw Pixel Encoder
 *
 * The Raw Pixel Encoder takes an array of images (as PHP Resources)
 * and converts them into a flat vector of raw color channel data. Scaling and
 * cropping is handled automatically by Intervention Image for PHP. Note that
 * the GD extension is required to use this feature.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RawPixelEncoder implements Extractor
{
    /**
     * The image will be scaled and cropped according to the setting of this
     * parameter which will have an effect on the size of the outpput vector.
     *
     * @var array
     */
    protected $size;

    /**
     * The number of color channels to encode.
     *
     * @var int
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
    public function __construct(array $size = [32, 32], bool $rgb = true, string $driver = 'gd')
    {
        if (count($size) !== 2) {
            throw new InvalidArgumentException('Image size must contain both'
                . ' width and height.');
        }

        if (!is_int($size[0]) and !is_int($size[1])) {
            throw new InvalidArgumentException('Image width and height must be'
                . ' integers.');
        }

        if ($size[0] < 1 or $size[1] < 1) {
            throw new InvalidArgumentException('Image width and height must be'
                . ' greater than 1 pixel.');
        }

        $this->size = $size;
        $this->channels = $rgb ? 3 : 1;
        $this->intervention = new ImageManager(['driver' => $driver]);
    }

    /**
     * Return the dimensionality of the vector that gets encoded.
     *
     * @return int
     */
    public function dimensions() : int
    {
        return (int) ($this->size[0] * $this->size[1] * $this->channels);
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

                $image = $image->fit(...$this->size)->getCore();

                $vectors[] = $this->vectorize($image);
            }
        }

        return $vectors;
    }

    /**
     * Convert an image into a vector of raw color channel data.
     *
     * @param  resource  $image
     * @throws \InvalidArgumentException
     * @return array
     */
    public function vectorize($image) : array
    {
        if (!is_resource($image)) {
            throw new InvalidArgumentException('Input is not a resource.');
        }

        list($width, $height) = $this->size;

        $vector = [];

        for ($x = 0; $x < $width; $x++) {
            for ($y = 0; $y < $height; $y++) {
                $pixels = imagecolorsforindex($image, imagecolorat($image, $x, $y));

                $pixels = array_slice($pixels, 0, $this->channels);

                $vector = array_merge($vector, array_values($pixels));
            }
        }

        return $vector;
    }
}
