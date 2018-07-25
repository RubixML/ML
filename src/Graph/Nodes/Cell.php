<?php

namespace Rubix\ML\Graph\Nodes;

class Cell extends BinaryNode
{
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
    public function __construct(float $score)
    {
        $this->score = $score;
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
