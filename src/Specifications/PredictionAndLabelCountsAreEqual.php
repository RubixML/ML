<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * @internal
 */
class PredictionAndLabelCountsAreEqual
{
    /**
     * The predictions returned from an estimator.
     *
     * @var (string|int|float)[]
     */
    protected array $predictions;

    /**
     * The ground-truth labels.
     *
     * @var (string|int|float)[]
     */
    protected array $labels;

    /**
     * Build a specification object with the given arguments.
     *
     * @param (string|int|float)[] $predictions
     * @param (string|int|float)[] $labels
     * @return self
     */
    public static function with(array $predictions, array $labels) : self
    {
        return new self($predictions, $labels);
    }

    /**
     * @param (string|int|float)[] $predictions
     * @param (string|int|float)[] $labels
     */
    public function __construct(array $predictions, array $labels)
    {
        $this->predictions = $predictions;
        $this->labels = $labels;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function check() : void
    {
        if (count($this->predictions) !== count($this->labels)) {
            throw new InvalidArgumentException(
                'Number of predictions and labels must be equal '
                . count($this->predictions) . ' predictions but '
                . count($this->labels) . ' labels given.'
            );
        }
    }
}
