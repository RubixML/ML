<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Kernels\SVM;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Kernels\SVM\Sigmoidal;
use PHPUnit\Framework\TestCase;

#[Group('Kernels')]
#[RequiresPhpExtension('svm')]
#[CoversClass(Sigmoidal::class)]
class SigmoidalTest extends TestCase
{
    protected Sigmoidal $kernel;

    protected function setUp() : void
    {
        $this->kernel = new Sigmoidal(gamma: 1e-3);
    }

    public function testOptions() : void
    {
        $options = [
            102 => 3,
            201 => 1e-3,
            205 => 0.0,
        ];

        $this->assertEquals($options, $this->kernel->options());
    }
}
