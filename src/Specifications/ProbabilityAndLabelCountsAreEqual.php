<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * @internal
 */
class ProbabilityAndLabelCountsAreEqual
{
    /**
     * The probabilities returned from an estimator.
     *
     * @var list<array<string|int,float>>
     */
    protected array $probabilities;

    /**
     * The ground-truth labels.
     *
     * @var (string|int)[]
     */
    protected array $labels;

    /**
     * Build a specification object with the given arguments.
     *
     * @param list<array<string|int,float>> $probabilities
     * @param (string|int)[] $labels
     * @return self
     */
    public static function with(array $probabilities, array $labels) : self
    {
        return new self($probabilities, $labels);
    }

    /**
     * @param list<array<string|int,float>> $probabilities
     * @param (string|int)[] $labels
     */
    public function __construct(array $probabilities, array $labels)
    {
        $this->probabilities = $probabilities;
        $this->labels = $labels;
    }

    /**
     * Perform a check of the specification and throw an exception if invalid.
     *
     * @throws InvalidArgumentException
     */
    public function check() : void
    {
        if (count($this->probabilities) !== count($this->labels)) {
            throw new InvalidArgumentException(
                'Number of probabilities and labels must be equal '
                . count($this->probabilities) . ' predictions but '
                . count($this->labels) . ' labels given.'
            );
        }
    }
}
