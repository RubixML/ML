<?php

namespace Rubix\ML\Graph\Nodes;

/**
 * Cell
 *
 * A cell node contains samples that are likely members of the same
 * group.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Cell extends BinaryNode implements Leaf
{
    /**
     * The number of training points located in this cell.
     *
     * @var int
     */
    protected $n;

    /**
     * The isolation score.
     *
     * @var float
     */
    protected $score;

    /**
     * @param  float  $score
     * @return void
     */
    public function __construct(int $n, float $score)
    {
        $this->n = $n;
        $this->score = $score;
    }

    /**
     * Return the number of training points located in this cell.
     *
     * @return int
     */
    public function n() : int
    {
        return $this->n;
    }

    /**
     * Return the isloation score.
     *
     * @return float
     */
    public function score() : float
    {
        return $this->score;
    }
}
