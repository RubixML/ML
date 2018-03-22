<?php

namespace Rubix\Engine\Transformers;

class Splitter
{
    /**
     * The training data.
     *
     * @var array
     */
    protected $training;

    /**
     * The testing data.
     *
     * @var array
     */
    protected $testing;

    /**
     * @param  array  $data
     * @param  float  $ratio
     * @return void
     */
    public function __construct(array $data = [], float $ratio = 0.3)
    {
        if ($ratio <= 0.0 || $ratio >= 0.9) {
            throw new InvalidArgumentException('Split ratio must be a float value between 0.0 and 0.9.');
        }

        list($this->training, $this->testing) = $this->split($data, $ratio);
    }

    /**
     * @param  array  $data
     * @param  float  $ratio
     * @return array
     */
    protected function split(array $data, float $ratio) : array
    {
        $testing = array_splice($data, floor($ratio * count($data)));

        return [$data, $testing];
    }

    /**
     * @return array
     */
    public function training() : array
    {
        return $this->training;
    }

    /**
     * @return array
     */
    public function testing() : array
    {
        return $this->testing;
    }

    /**
     * @return float
     */
    public function ratio() : float
    {
        return count($this->testing) / $this->total();
    }

    /**
     * @return int
     */
    public function total() : int
    {
        return count($this->training) + count($this->testing);
    }
}
