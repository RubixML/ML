<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function rand;
use function array_walk;
use function getrandmax;

/**
 * Randomized Image Flipper
 *
 * Randomly flips images along the horizontal and/or vertical axis with a given probability. Flipping is
 * lossless since the pixel values are only permuted, making it a common augmentation for training computer
 * vision models on data such as digits, faces, or natural scenes.
 *
 * > **Note**: The GD extension is required to use this transformer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ImageFlipper implements Transformer
{
    /**
     * The probability of flipping an image along the horizontal axis.
     *
     * @var float
     */
    protected float $horizontal;

    /**
     * The probability of flipping an image along the vertical axis.
     *
     * @var float
     */
    protected float $vertical;

    /**
     * @param float $horizontal
     * @param float $vertical
     * @throws InvalidArgumentException
     */
    public function __construct(float $horizontal = 0.5, float $vertical = 0.0)
    {
        ExtensionIsLoaded::with('gd')->check();

        if ($horizontal < 0.0 or $horizontal > 1.0) {
            throw new InvalidArgumentException('Horizontal probability must be'
                . " between 0 and 1 and $horizontal given.");
        }

        if ($vertical < 0.0 or $vertical > 1.0) {
            throw new InvalidArgumentException('Vertical probability must be'
                . " between 0 and 1 and $vertical given.");
        }

        $this->horizontal = $horizontal;
        $this->vertical = $vertical;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<DataType>
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Transform the dataset in place.
     *
     * @param array<mixed[]> $samples
     */
    public function transform(array &$samples) : void
    {
        array_walk($samples, [$this, 'flip']);
    }

    /**
     * Randomly flip the images in a sample along each axis.
     *
     * @internal
     *
     * @param list<mixed> $sample
     * @throws RuntimeException
     */
    protected function flip(array &$sample) : void
    {
        foreach ($sample as &$value) {
            if (DataType::detect($value)->isImage()) {
                if ($this->shouldFlip($this->horizontal) and !imageflip($value, IMG_FLIP_HORIZONTAL)) {
                    throw new RuntimeException('Failed to flip image horizontally.');
                }

                if ($this->shouldFlip($this->vertical) and !imageflip($value, IMG_FLIP_VERTICAL)) {
                    throw new RuntimeException('Failed to flip image vertically.');
                }
            }
        }

        unset($value);
    }

    /**
     * Determine if an axis should be flipped given its probability.
     *
     * @internal
     *
     * @param float $probability
     * @return bool
     */
    protected function shouldFlip(float $probability) : bool
    {
        if ($probability <= 0.0) {
            return false;
        }

        if ($probability >= 1.0) {
            return true;
        }

        return rand() / getrandmax() < $probability;
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "Image Flipper (horizontal: {$this->horizontal}, vertical: {$this->vertical})";
    }
}
