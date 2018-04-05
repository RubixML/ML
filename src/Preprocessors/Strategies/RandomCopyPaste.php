<?php

namespace Rubix\Engine\Preprocessors\Strategies;

class RandomCopyPaste implements Categorical, Continuous
{
    /**
     * Copy and paste a random value from the data.
     *
     * @param  array  $values
     * @return mixed
     */
    public function guess(array $values)
    {
        return $values[array_rand($values)];
    }
}
