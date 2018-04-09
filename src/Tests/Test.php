<?php

namespace Rubix\Engine\Tests;

interface Test
{
    /**
     * Run the test.
     *
     * @param  array  $predictions
     * @param  array|null  $outcomes
     * @return bool
     */
    public function test(array $predictions, ?array $outcomes = null) : bool;
}
