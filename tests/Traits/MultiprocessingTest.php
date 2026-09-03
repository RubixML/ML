<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Traits;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Traits\Multiprocessing;
use PHPUnit\Framework\TestCase;

#[Group('Traits')]
#[CoversClass(Multiprocessing::class)]
class MultiprocessingTest extends TestCase
{
    protected MultiprocessingFixture $fixture;

    protected function setUp() : void
    {
        $this->fixture = new MultiprocessingFixture();
    }

    #[Test]
    public function backendIsNullByDefault() : void
    {
        $this->assertNull($this->fixture->backend());
    }

    #[Test]
    public function setBackendAcceptsAConcreteBackend() : void
    {
        $backend = new Serial();

        $this->fixture->setBackend($backend);

        $this->assertSame($backend, $this->fixture->backend());
    }
}

/**
 * A fixture exposing the Multiprocessing trait.
 *
 * @internal
 */
class MultiprocessingFixture
{
    use Multiprocessing;

    public function backend() : ?Backend
    {
        return $this->backend;
    }
}
