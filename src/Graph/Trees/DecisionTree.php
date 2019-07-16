<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Graph\Nodes\Outcome;

interface DecisionTree extends BinaryTree
{
    /**
     * Search the decision tree for a leaf node and return it.
     *
     * @param array $sample
     * @return \Rubix\ML\Graph\Nodes\Outcome|null
     */
    public function search(array $sample) : ?Outcome;

    /**
     * Return an array indexed by feature column that contains the normalized
     * importance score of that feature.
     *
     * @return array
     */
    public function featureImportances() : array;

    /**
     * Print a human readable text representation of the decision tree.
     */
    public function printRules() : void;
}
