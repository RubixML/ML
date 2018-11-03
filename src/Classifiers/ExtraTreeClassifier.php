<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Comparison;
use InvalidArgumentException;

/**
 * Extra Tree Classifier
 *
 * An Extremely Randomized Classification Tree that splits the training set at
 * a random point chosen among the maximum features. Extra Trees are useful in
 * Ensembles such as Random Forest or AdaBoost as the “weak” classifier or they
 * can be used on their own. The strength of Extra Trees are computational
 * efficiency as well as increasing variance of the prediction (if that is
 * desired).
 *
 * References:
 * [1] P. Geurts et al. (2005). Extremely Randomized Trees.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ExtraTreeClassifier extends ClassificationTree
{
    /**
     * Randomized algorithm to that choses the split point with the lowest gini
     * impurity among a random selection of $maxFeatures features.
     *
     * @param  \Rubix\ML\Datasets\Labeled  $dataset
     * @param  int  $depth
     * @return \Rubix\ML\Graph\Nodes\Comparison
     */
    protected function findBestSplit(Labeled $dataset, int $depth) : Comparison
    {
        $bestGini = INF;
        $bestIndex = $bestValue = null;
        $bestGroups = [];

        $max = $dataset->numRows() - 1;

        shuffle($this->indices);

        foreach (array_slice($this->indices, 0, $this->maxFeatures) as $index) {
            $sample = $dataset->row(rand(0, $max));

            $value = $sample[$index];

            $groups = $dataset->partition($index, $value);

            $gini = $this->gini($groups);

            if ($gini < $bestGini) {
                $bestValue = $value;
                $bestIndex = $index;
                $bestGroups = $groups;
                $bestGini = $gini;
            }

            if ($gini <= $this->tolerance) {
                break 1;
            }
        }

        if ($this->logger) $this->logger->info("Best split: column=$bestIndex"
            . " value=$bestValue impurity=$bestGini depth=$depth");

        return new Comparison($bestValue, $bestIndex, $bestGroups, $bestGini);
    }
}
