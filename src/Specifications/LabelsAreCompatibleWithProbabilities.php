<?php

namespace Rubix\ML\Specifications;

use Rubix\ML\Exceptions\InvalidArgumentException;

use function count;

/**
 * @internal
 */
class LabelsAreCompatibleWithProbabilities extends Specification
{
    /**
     * Predicted probabilities.
     *
     * @var list<array<float>>
     */
    protected array $probabilities;

    /**
     * Dataset labels.
     *
     * @var list<string|int>
     */
    protected array $labels;

    /**
     * Build a specification object with the given arguments.
     *
     * @param list<array<float>> $probabilities
     * @param list<string|int> $labels
     * @return self
     */
    public static function with(array $probabilities, array $labels) : self
    {
        return new self($probabilities, $labels);
    }

    /**
     * @param list<array<float>> $probabilities
     * @param list<string|int> $labels
     */
    public function __construct(array $probabilities, array $labels)
    {
        $this->probabilities = $probabilities;
        $this->labels = $labels;
    }

    /**
     * Perform a check of the specification.
     *
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function check() : void
    {
        $countProbabilities = count($this->probabilities);
        $countLabels = count($this->labels);

        if ($countProbabilities != $countLabels) {
            throw new InvalidArgumentException(
                'Labels are incompatible with predictions'
                . "($countLabels labels and $countProbabilities probabilities provided)."
            );
        }
    }
}
