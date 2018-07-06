<?php

namespace Rubix\ML\Graph\Nodes;

class Decision extends BinaryNode
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
     * Remove the left and right splits of training data.
     *
     * @return void
     */
    public function cleanup() : void
    {
        unset($this->groups);
    }
}
