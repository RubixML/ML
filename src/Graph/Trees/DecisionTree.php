<?php

namespace Rubix\ML\Graph\Trees;

interface DecisionTree extends BST
{
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
