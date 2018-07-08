<?php

namespace Rubix\ML\Classifiers;

use Rubix\ML\Graph\Nodes\Decision;
use InvalidArgumentException;

class ExtraTree extends ClassificationTree
{
    /**
     * Randomized algorithm to that choses the split point with the lowest gini
     * impurity among a random selection of $maxFeatures features.
     *
     * @param  array  $data
     * @return \Rubix\ML\Graph\Nodes\Decision
     */
    protected function findBestSplit(array $data) : Decision
    {
        $best = [
            'gini' => INF, 'index' => null, 'value' => null, 'groups' => [],
        ];

        shuffle($this->indices);

        foreach (array_slice($this->indices, 0, $this->maxFeatures) as $index) {
            $value = $data[random_int(0, count($data) - 1)][$index];

            $groups = $this->partition($data, $index, $value);

            $gini = $this->calculateGiniImpurity($groups);

            if ($gini < $best['gini']) {
                $best['gini'] = $gini;
                $best['index'] = $index;
                $best['value'] = $value;
                $best['groups'] = $groups;
            }

            if ($gini === 0.0) {
                break 1;
            }
        }

        return new Decision($best['index'], $best['value'],
            $best['gini'], $best['groups']);
    }
}
