<?php

namespace Rubix\ML\Extractors;

use Intervention\Image\Image;
use Intervention\Image\ImageManager;
use Rubix\ML\Extractors\Descriptors\Descriptor;
use InvalidArgumentException;
use RuntimeException;

/**
 * Image Patch Descriptor
 *
 * This image extractor encodes various user-defined features called descriptors
 * using subsamples of the original image called *patches*.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ImagePatchDescriptor implements Extractor
{
    /**
     * The descriptor middelware. Each descriptor encodes a set of features
     * from a patch of the image.
     *
     * @var array
     */
    protected $descriptors;

    /**
     * The image will be scaled and cropped according to the setting of this
     * parameter which will have an effect on the size of the outpput vector.
     *
     * @var array
     */
    protected $size;

    /**
     * The width and height of each patch i.e. a subsampling of the full image.
     *
     * @var array
     */
    protected $patchSize;

    /**
     * The Intervention image manager instance.
     *
     * @var \Intervention\Image\ImageManager
     */
    protected $intervention;

    /**
     * @param  array  $descriptors
     * @param  array  $size
     * @param  array  $patchSize
     * @param  string  $driver
     * @throws \InvalidArgumentException
     * @return void
     */
    public function __construct(array $descriptors, array $size = [32, 32], array $patchSize = [4, 4],
                                string $driver = 'gd')
    {
        if (count($descriptors) === 0) {
            throw new InvalidArgumentException('Must provide at least 1'
                . ' descriptor.');
        }

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

        if (count($patchSize) !== 2) {
            throw new InvalidArgumentException('Patch size must contain both'
                . ' width and height.');
        }

        if (!is_int($patchSize[0]) and !is_int($patchSize[1])) {
            throw new InvalidArgumentException('Patch width and height must be'
                . ' integers.');
        }

        if ($patchSize[0] < 1 or $patchSize[1] < 1) {
            throw new InvalidArgumentException('Patch width and height must be'
                . ' greater than 1 pixel.');
        }

        if ($patchSize[0] > $size[0] or $patchSize[1] > $size[1]) {
            throw new InvalidArgumentException('Patch size cannot be greater'
                . ' than image size.');
        }


        foreach ($descriptors as $descriptor) {
            $this->addDescriptor($descriptor);
        }

        $this->size = $size;
        $this->patchSize = $patchSize;
        $this->intervention = new ImageManager(['driver' => $driver]);
    }

    /**
     * Return the number of patches that are described by this extractor.
     *
     * @return int
     */
    public function numPatches() : int
    {
        return (int) (floor($this->size[0] / $this->patchSize[0])
            * floor($this->size[1] / $this->patchSize[1]));
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

                $image = $image->fit(...$this->size)->getCore();

                $vectors[] = $this->vectorize($image);
            }
        }

        return $vectors;
    }

    /**
     * Extract patches from a given image and run descriptor middleware over
     * them in order from top left to bottom right.
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

        list($imageWidth, $imageHeight) = $this->size;

        list($patchWidth, $patchHeight) = $this->patchSize;

        $nX = (int) floor($imageWidth / $patchWidth);
        $nY = (int) floor($imageHeight / $patchHeight);

        $vector = [];

        for ($xHat = 0; $xHat < $nX; $xHat++) {
            for ($yHat = 0; $yHat < $nY; $yHat++) {
                $xStart = $xHat * $patchWidth;
                $yStart = $yHat * $patchHeight;

                $xEnd = $xStart + $patchWidth;
                $yEnd = $yStart + $patchHeight;

                $patch = [[]];

                for ($x = $xStart; $x < $xEnd; $x++) {
                    for ($y = $yStart; $y < $yEnd; $y++) {
                        $rgba = imagecolorat($image, $x, $y);

                        $pixels = imagecolorsforindex($image, $rgba);

                        $pixels = array_slice($pixels, 0, 3, true);

                        $patch[$x][$y] = $pixels;
                    }
                }

                foreach ($this->descriptors as $descriptor) {
                    $features = $descriptor->describe($patch);

                    $vector = array_merge($vector, $features);
                }
            }
        }

        return $vector;
    }

    /**
     * Add a transformer middleware to the pipeline.
     *
     * @param  \Rubix\ML\Extractors\Descriptors\Descriptor  $descriptor
     * @return void
     */
    protected function addDescriptor(Descriptor $descriptor) : void
    {
        $this->descriptors[] = $descriptor;
    }
}
