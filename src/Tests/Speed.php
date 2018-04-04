<?php

namespace Rubix\Engine\Tests;

use MathPHP\Statistics\Average;

class Speed extends Test
{
    /**
     * The minimum speed score in seconds.
     *
     * @var int
     */
    protected $seconds;

    /**
     * The decimal precision of the speed measurement.
     *
     * @var int
     */
    protected $precision;

    /**
     * @param  float  $seconds
     * @param  int  $precision
     * @return void
     */
    public function __construct(float $seconds = 0.3, int $precision = 5)
    {
        $this->seconds = $seconds;
        $this->precision = $precision;
    }

    /**
     * Test the speed of the estimator.
     *
     * @param  array  $samples
     * @param  array  $outcomes
     * @return bool
     */
    public function test(array $samples, array $outcomes) : bool
    {
        $speeds = [];

        foreach ($samples as $sample) {
            $start = microtime(true);

            $this->estimator->predict($sample);

            $speeds[] = round(microtime(true) - $start, $this->precision);
        }

        $average = round(Average::mean($speeds), $this->precision);

        $pass = $average <= $this->seconds;

        echo 'Predictions took ' . (string) $average . 's on average - ' . ($pass ? 'PASS' : 'FAIL') . "\n";

        return $pass;
    }
}
