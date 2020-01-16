<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Specifications\SamplesAreCompatibleWithTransformer;
use RuntimeException;

use function is_null;

/**
 * Image Vectorizer
 *
 * Image Vectorizer takes images of the same size and converts them into flat feature vectors
 * of raw color channel intensities. Intensities range from 0 to 255 and can either be read
 * from 1 channel (grayscale) or 3 channels (RGB color) per pixel.
 *
 * > **Note**: The GD extension is required to use this transformer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ImageVectorizer implements Transformer, Stateful
{
    /**
     * The number of color channels to encode.
     *
     * @var int
     */
    protected $channels;

    /**
     * The the number of bits to encode after the first 8.
     *
     * @var int
     */
    protected $mu;

    /**
     * The fixed width and height of the images for each image feature column.
     *
     * @var array[]|null
     */
    protected $sizes;

    /**
     * @param bool $grayscale
     * @throws \InvalidArgumentException
     * @throws \RuntimeException
     */
    public function __construct(bool $grayscale = false)
    {
        if (!extension_loaded('gd')) {
            throw new RuntimeException('GD extension is not loaded, check'
                . ' PHP configuration.');
        }

        $channels = $grayscale ? 1 : 3;

        $this->channels = $channels;
        $this->mu = ($channels - 1) * 8;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return int[]
     */
    public function compatibility() : array
    {
        return DataType::ALL;
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->sizes);
    }

    /**
     * Fit the transformer to the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::check($dataset, $this);

        $sample = $dataset->sample(0);

        $this->sizes = [];

        foreach ($dataset->types() as $column => $type) {
            if ($type === DataType::IMAGE) {
                $value = $sample[$column];

                $this->sizes[$column] = [
                    imagesx($value),
                    imagesy($value),
                ];
            }
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->sizes)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            $vectors = [];

            foreach ($this->sizes as $column => [$width, $height]) {
                $value = $sample[$column];

                $vector = [];

                for ($x = 0; $x < $width; ++$x) {
                    for ($y = 0; $y < $height; ++$y) {
                        $pixel = imagecolorat($value, $x, $y);

                        $vector[] = $pixel & 0xFF;
        
                        for ($i = 8; $i <= $this->mu; $i *= 2) {
                            $vector[] = ($pixel >> $i) & 0xFF;
                        }
                    }
                }

                unset($sample[$column]);

                imagedestroy($value);

                $vectors[] = $vector;
            }

            $sample = array_merge($sample, ...$vectors);
        }
    }
}
