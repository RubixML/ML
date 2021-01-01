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
     * @var \Rubix\ML\Datasets\Dataset
     */
    protected $dataset;

    /**
     * The transformer.
     *
     * @var \Rubix\ML\Transformers\Transformer
     */
    protected $transformer;

    /**
     * Build a specification object with the given arguments.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Transformers\Transformer $transformer
     * @return self
     */
    public static function with(Dataset $dataset, Transformer $transformer) : self
    {
        return new self($dataset, $transformer);
    }

    /**
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Transformers\Transformer $transformer
     */
    public function __construct(Dataset $dataset, Transformer $transformer)
    {
        $this->dataset = $dataset;
        $this->transformer = $transformer;
    }

    /**
     * Perform a check of the specification.
     *
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
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
