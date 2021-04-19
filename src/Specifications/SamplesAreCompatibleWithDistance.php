<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Kernels\Distance\Distance;
use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;

/**
 * @internal
 */
class SamplesAreCompatibleWithDistance extends Specification
{
    /**
     * The dataset that contains samples under validation.
     *
     * @var \Rubix\ML\Datasets\Dataset
     */
    protected \Rubix\ML\Datasets\Dataset $dataset;

    /**
     * The distance kernel.
     *
     * @var \Rubix\ML\Kernels\Distance\Distance
     */
    protected \Rubix\ML\Kernels\Distance\Distance $kernel;

    /**
     * Build a specification object with the given arguments.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @return self
     */
    public static function with(Dataset $dataset, Distance $kernel) : self
    {
        return new self($dataset, $kernel);
    }

    /**
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     */
    public function __construct(Dataset $dataset, Distance $kernel)
    {
        $this->dataset = $dataset;
        $this->kernel = $kernel;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function check() : void
    {
        $compatibility = $this->kernel->compatibility();

        $types = $this->dataset->uniqueTypes();

        $compatible = array_intersect($types, $compatibility);

        if (count($compatible) < count($types)) {
            $incompatible = array_diff($types, $compatibility);

            throw new InvalidArgumentException(
                "{$this->kernel} is incompatible with " . implode(', ', $incompatible) . ' data types.'
            );
        }
    }
}
