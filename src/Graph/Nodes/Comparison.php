<?php

namespace Rubix\ML\Graph\Nodes;

class Comparison extends BinaryNode
{
    /**
     * The index of the feature column.
     *
     * @var int
     */
    protected $index;

    /**
     * The split value.
     *
     * @var mixed
     */
    protected $value;

    /**
     * The score of the decision. i.e. the amount of gini impurity or sse that
     * the split introduces.
     *
     * @var float
     */
    protected $score;

    /**
     * The left and right splits of the training data.
     *
     * @var array
     */
    protected $groups = [
        //
    ];

    /**
     * The number of training samples this node is responsible for.
     *
     * @var int
     */
    protected $n;

    /**
     * @param  int  $index
     * @param  mixed  $value
     * @param  float  $score
     * @param  array  $groups
     * @return void
     */
    public function __construct(int $index, $value, float $score, array $groups)
    {
        $this->index = $index;
        $this->value = $value;
        $this->score = $score;
        $this->n = array_sum(array_map('count', $groups));
        $this->groups = $groups;
    }

    /**
     * @return int
     */
    public function index() : int
    {
        return $this->index;
    }

    /**
     * @return mixed
     */
    public function value()
    {
        return $this->value;
    }

    /**
     * @return float
     */
    public function score() : float
    {
        return $this->score;
    }

    /**
     * @return array
     */
    public function groups() : array
    {
        return $this->groups;
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

    /**
     * Remove the left and right splits of training data.
     *
     * @return void
     */
    public function cleanup() : void
    {
        unset($this->groups);
    }
}
