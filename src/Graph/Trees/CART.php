<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Helpers\Stats;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Graph\Nodes\Decision;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function Rubix\ML\linspace;
use function count;
use function round;
use function log;
use function sqrt;
use function array_slice;
use function array_unique;
use function array_rand;

/**
 * CART
 *
 * *Classification and Regression Tree* or CART is a binary search tree that uses *decision* nodes
 * at every split in the training data to locate a purified leaf node.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
abstract class CART extends DecisionTree
{
    /**
     * The maximum number of features to consider when determining a split.
     *
     * @var int|null
     */
    protected ?int $maxFeatures = null;

    /**
     * The maximum number of bins to consider when determining a split with a continuous feature
     * as the split point.
     *
     * @var int|null
     */
    protected ?int $maxBins = null;

    /**
     * @internal
     *
     * @param int $maxHeight
     * @param int $maxLeafSize
     * @param float $minPurityIncrease
     * @param int|null $maxFeatures
     * @param int|null $maxBins
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $maxHeight,
        int $maxLeafSize,
        float $minPurityIncrease,
        ?int $maxFeatures,
        ?int $maxBins
    ) {
        parent::__construct($maxHeight, $maxLeafSize, $minPurityIncrease);

        if (isset($maxFeatures) and $maxFeatures < 1) {
            throw new InvalidArgumentException('Tree must consider at least 1'
                . " feature to determine a split, $maxFeatures given.");
        }

        if (isset($maxBins) and $maxBins < 2) {
            throw new InvalidArgumentException('At least two bins are'
                . " required to determine a split, $maxBins given.");
        }

        $this->maxFeatures = $maxFeatures;
        $this->maxBins = $maxBins;
    }

    /**
     * Greedy algorithm to choose the best split point for a given dataset.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return \Rubix\ML\Graph\Nodes\Split
     */
    protected function split(Labeled $dataset) : Split
    {
        [$m, $n] = $dataset->shape();

        $maxFeatures = $this->maxFeatures ?? (int) round(sqrt($n));
        $bins = $this->maxBins ?? 1 + (int) round(log($m, 2));

        $columns = array_fill(0, $dataset->numFeatures(), null);

        $columns = (array) array_rand($columns, min($maxFeatures, count($columns)));

        $bestColumn = $bestValue = $bestSubsets = null;
        $bestImpurity = INF;

        foreach ($columns as $column) {
            $type = $dataset->featureType($column);
            $values = $dataset->feature($column);

            $values = array_unique($values);

            if ($type->isContinuous()) {
                if (count($values) > $bins) {
                    if (!isset($q)) {
                        $q = linspace(0.0, 1.0, $bins + 1);

                        $q = array_slice($q, 1, -1);
                    }

                    $values = Stats::quantiles($values, $q);
                }
            } else {
                if (count($values) === 2) {
                    $values = array_slice($values, 0, 1);
                }
            }

            foreach ($values as $value) {
                $subsets = $dataset->splitByFeature($column, $value);

                $impurity = $this->splitImpurity($subsets);

                if ($impurity < $bestImpurity) {
                    $bestColumn = $column;
                    $bestValue = $value;
                    $bestSubsets = $subsets;
                    $bestImpurity = $impurity;
                }

                if ($impurity <= 0.0) {
                    break 2;
                }
            }
        }

        if ($bestColumn === null or $bestValue === null or $bestSubsets === null) {
            throw new RuntimeException('Could not split dataset.');
        }

        return new Split(
            $bestColumn,
            $bestValue,
            $bestSubsets,
            $bestImpurity,
            $m
        );
    }
}
