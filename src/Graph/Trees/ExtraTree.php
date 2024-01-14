<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function is_int;
use function array_fill;
use function array_unique;
use function array_rand;
use function floor;
use function ceil;
use function max;
use function abs;
use function getrandmax;
use function rand;

/**
 * Extra Tree
 *
 * The base implementation of an *Extremely Randomized* Decision Tree.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
abstract class ExtraTree extends DecisionTree
{
    /**
     * The maximum number of features to consider when determining a split.
     *
     * @var int|null
     */
    protected ?int $maxFeatures = null;

    /**
     * @internal
     *
     * @param int $maxHeight
     * @param int $maxLeafSize
     * @param float $minPurityIncrease
     * @param int|null $maxFeatures
     * @throws \InvalidArgumentException
     */
    public function __construct(
        int $maxHeight,
        int $maxLeafSize,
        float $minPurityIncrease,
        ?int $maxFeatures
    ) {
        parent::__construct($maxHeight, $maxLeafSize, $minPurityIncrease);

        if (isset($maxFeatures) and $maxFeatures < 1) {
            throw new InvalidArgumentException('Tree must consider at least 1'
                . " feature to determine a split, $maxFeatures given.");
        }

        $this->maxFeatures = $maxFeatures;
    }

    /**
     * Randomized algorithm that chooses the split point with the lowest impurity
     * among a random selection of features.
     *
     * @param Labeled $dataset
     * @return Split
     */
    protected function split(Labeled $dataset) : Split
    {
        [$m, $n] = $dataset->shape();

        $maxFeatures = $this->maxFeatures ?? (int) round(sqrt($n));

        $columns = array_fill(0, $dataset->numFeatures(), null);

        $columns = (array) array_rand($columns, min($maxFeatures, count($columns)));

        $randMax = getrandmax();

        $bestColumn = $bestValue = $bestGroups = null;
        $bestImpurity = INF;

        foreach ($columns as $column) {
            $values = $dataset->feature($column);

            $type = $dataset->featureType($column);

            if ($type->isContinuous()) {
                $min = min($values);
                $max = max($values);

                $maxAbs = max(abs($max), abs($min));

                $phi = $maxAbs != 0.0 ? $randMax / $maxAbs : $randMax;

                $min = (int) floor($min * $phi);
                $max = (int) ceil($max * $phi);

                $value = rand($min, $max) / $phi;
            } else {
                $values = array_unique($values);

                $offset = array_rand($values);

                $value = $values[$offset];
            }

            $groups = $dataset->splitByFeature($column, $value);

            $impurity = $this->splitImpurity($groups);

            if ($impurity < $bestImpurity) {
                $bestColumn = $column;
                $bestValue = $value;
                $bestGroups = $groups;
                $bestImpurity = $impurity;
            }

            if ($impurity <= 0.0) {
                break;
            }
        }

        if (!is_int($bestColumn) or $bestValue === null or $bestGroups === null) {
            throw new RuntimeException('Could not split dataset.');
        }

        return new Split(
            $bestColumn,
            $bestValue,
            $bestGroups,
            $bestImpurity,
            $m
        );
    }
}
