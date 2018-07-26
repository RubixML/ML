<?php

namespace Rubix\ML\Graph\Nodes;

class Comparison extends Split
{
    /**
     * The score of the decision. i.e. the amount of gini impurity or sse that
     * the split introduces.
     *
     * @var float
     */
    protected $score;

    /**
     * The number of training samples this node is responsible for.
     *
     * @var int
     */
    protected $n;

    /**
     * @param  int  $index
     * @param  mixed  $value
     * @param  array  $groups
     * @param  float  $score
     * @return void
     */
    public function __construct(int $index, $value, array $groups, float $score)
    {
        $this->score = $score;
        $this->n = array_sum(array_map('count', $groups));

        parent::__construct($index, $value, $groups);
    }

    /**
     * @return float
     */
    public function score() : float
    {
        return $this->score;
    }

    /**
     * @return int
     */
    public function n() : int
    {
        return $this->n;
    }

    /**
     * Return the  decearse in impurity this decision node provides. A negative
     * score means that the decision node actually causes its children to become
     * less pure.
     *
     * @return float
     */
    public function impurityDecrease() : float
    {
        $decrease = $this->score;

        if (isset($this->left)) {
            if ($this->left instanceof Comparison) {
                $decrease -= $this->left->score()
                    * ($this->left->n() / $this->n);
            }
        }

        if (isset($this->right)) {
            if ($this->right instanceof Comparison) {
                $decrease -= $this->right->score()
                    * ($this->right->n() / $this->n);
            }
        }

        return $decrease;
    }
}
