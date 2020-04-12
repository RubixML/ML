<?php

namespace Rubix\ML\Graph\Trees;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Graph\Nodes\Outcome;

interface DecisionTree extends BinaryTree
{
    /**
     * Insert a root node and recursively split the dataset until a
     * terminating condition is met.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     */
    public function grow(Labeled $dataset) : void;

    /**
     * Search the decision tree for a leaf node and return it.
     *
     * @param (string|int|float)[] $sample
     * @return \Rubix\ML\Graph\Nodes\Outcome|null
     */
    public function search(array $sample) : ?Outcome;

    /**
     * Return an array indexed by feature column that contains the normalized
     * importance score of that feature.
     *
     * @return (int|float)[]
     */
    public function featureImportances() : array;

    /**
     * Return a human readable text representation of the decision tree rules.
     *
     * @return string
     */
    public function rules() : string;
}
