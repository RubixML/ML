<?php

namespace Backends;

use Rubix\ML\Backends\Amp2;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Tests\Backends\AbstractAmp;

/**
 * @group Backends
 * @covers \Rubix\ML\Backends\Amp2
 * @requires function \Amp\async
 */
class Amp2Test extends AbstractAmp
{
    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->backend = new Amp2(4);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Amp2::class, $this->backend);
        $this->assertInstanceOf(Backend::class, $this->backend);
    }
}
