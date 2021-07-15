<?php

namespace Rubix\ML\CrossValidation\Metrics;

/**
 * An aggregate object representing the input data for Metric::score() method.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Alex Torchenko
 */
class ScoreInput
{
    /**
     * @var list<int|float|string>
     */
    private array $predictions;

    /**
     * @var list<int|float|string>
     */
    private array $labels;

    /**
     * @var list<array<string, int|float>>|null
     */
    private ?array $probabilities;

    /**
     * @param list<int|float|string> $predictions
     * @param list<int|float|string> $labels
     * @param list<array<string, int|float>>|null $probabilities
     */
    public function __construct(array $predictions, array $labels, ?array $probabilities = null)
    {
        $this->predictions = $predictions;
        $this->labels = $labels;
        $this->probabilities = $probabilities;
    }

    /**
     * @return list<int|float|string>
     */
    public function predictions() : array
    {
        return $this->predictions;
    }

    /**
     * @return list<int|float|string>
     */
    public function labels() : array
    {
        return $this->labels;
    }

    /**
     * @return list<array<string, int|float>>|null
     */
    public function probabilities() : ?array
    {
        return $this->probabilities;
    }
}
