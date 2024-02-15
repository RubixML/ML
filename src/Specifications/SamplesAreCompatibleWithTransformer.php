<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;

/**
 * @internal
 */
class SamplesAreCompatibleWithTransformer extends Specification
{
    /**
     * The dataset that contains samples under validation.
     *
     * @var Dataset
     */
    protected Dataset $dataset;

    /**
     * The transformer.
     *
     * @var Transformer
     */
    protected Transformer $transformer;

    /**
     * Build a specification object with the given arguments.
     *
     * @param Dataset $dataset
     * @param Transformer $transformer
     * @return self
     */
    public static function with(Dataset $dataset, Transformer $transformer) : self
    {
        return new self($dataset, $transformer);
    }

    /**
     * @param Dataset $dataset
     * @param Transformer $transformer
     */
    public function __construct(Dataset $dataset, Transformer $transformer)
    {
        $this->dataset = $dataset;
        $this->transformer = $transformer;
    }

    /**
     * Perform a check of the specification.
     *
     * @throws InvalidArgumentException
     */
    public function check() : void
    {
        $compatibility = $this->transformer->compatibility();

        $types = $this->dataset->uniqueTypes();

        $compatible = array_intersect($types, $compatibility);

        if (count($compatible) < count($types)) {
            $incompatible = array_diff($types, $compatibility);

            throw new InvalidArgumentException(
                "{$this->transformer} is incompatible with " . implode(', ', $incompatible) . ' data types.'
            );
        }
    }
}
