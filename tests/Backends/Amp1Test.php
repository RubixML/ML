<?php

namespace Backends;

use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Tests\Backends\AbstractAmp;

/**
 * @group Backends
 * @covers \Rubix\ML\Backends\Amp
 * @requires function \Amp\call
 */
class Amp1Test extends AbstractAmp
{
    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->backend = new Amp(4);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Amp::class, $this->backend);
        $this->assertInstanceOf(Backend::class, $this->backend);
    }
}
