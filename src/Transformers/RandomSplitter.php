<?php

namespace Rubix\Engine\Transformers;

class RandomSplitter extends Splitter
{
    /**
     * @param  array  $data
     * @param  float  $ratio
     * @return array
     */
    protected function split(array $data, float $ratio) : array
    {
        shuffle($data);

        return parent::split($data, $ratio);
    }
}
