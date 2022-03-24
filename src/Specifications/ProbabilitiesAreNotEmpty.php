<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;

/**
 * @internal
 */
class ProbabilitiesAreNotEmpty extends Specification
{
    /**
     * Predicted probabilities.
     *
     * @var list<array<float>>
     */
    protected array $probabilities;

    /**
     * Build a specification object with the given arguments.
     *
     * @param list<array<float>> $probabilities
     * @return self
     */
    public static function with(array $probabilities) : self
    {
        return new self($probabilities);
    }

    /**
     * @param list<array<float>> $probabilities
     */
    public function __construct(array $probabilities)
    {
        $this->probabilities = $probabilities;
    }

    /**
     * Perform a check of the specification.
     *
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function check() : void
    {
        if (empty($this->probabilities)) {
            throw new InvalidArgumentException(
                'Probabilities must be provided.'
            );
        }
    }
}
