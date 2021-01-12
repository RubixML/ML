<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Split;
use Rubix\ML\Exceptions\RuntimeException;

use function array_slice;

use const Rubix\ML\PHI;

/**
 * Extra Tree
 *
 * The base implementation of an *Extremely Randomized* Decision Tree.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
abstract class ExtraTree extends CART
{
    /**
     * Randomized algorithm that chooses the split point with the lowest impurity
     * among a random selection of features.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @return \Rubix\ML\Graph\Nodes\Split
     */
    protected function split(Labeled $dataset) : Split
    {
        $n = $dataset->numRows();

        $columns = array_keys($this->types);

        shuffle($columns);

        $columns = array_slice($columns, 0, $this->maxFeatures);

        $bestImpurity = INF;

        foreach ($columns as $column) {
            $values = $dataset->column($column);

            $type = $this->types[$column];

            if ($type->isContinuous()) {
                $min = (int) floor(min($values) * PHI);
                $max = (int) ceil(max($values) * PHI);

                $value = rand($min, $max) / PHI;
            } else {
                $offset = array_rand(array_unique($values));

                $value = $values[$offset];
            }

            $groups = $dataset->partitionByColumn($column, $value);

            $impurity = $this->splitImpurity($groups, $n);

            if ($impurity < $bestImpurity) {
                $node = new Split($column, $value, $groups, $impurity, $n);
            }

            if ($impurity <= 0.0) {
                break;
            }
        }

        if (!isset($node)) {
            throw new RuntimeException('Unable to split dataset.');
        }

        return $node;
    }
}
