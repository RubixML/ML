<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Traits;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Loggers\BlackHole;
use Rubix\ML\Loggers\Screen;
use Rubix\ML\Traits\LoggerAware;
use PHPUnit\Framework\TestCase;
use Psr\Log\LoggerInterface;

#[Group('Traits')]
#[CoversClass(LoggerAware::class)]
class LoggerAwareTest extends TestCase
{
    protected LoggerAwareFixture $fixture;

    protected function setUp() : void
    {
        $this->fixture = new LoggerAwareFixture();
    }

    #[Test]
    public function loggerIsNullByDefault() : void
    {
        $this->assertNull($this->fixture->logger());
    }

    #[Test]
    public function setLoggerAcceptsAConcreteLogger() : void
    {
        $logger = new BlackHole();

        $this->fixture->setLogger($logger);

        $this->assertInstanceOf(LoggerInterface::class, $this->fixture->logger());
        $this->assertSame($logger, $this->fixture->logger());
    }

    #[Test]
    public function setLoggerAcceptsScreen() : void
    {
        $logger = new Screen('testing');

        $this->fixture->setLogger($logger);

        $this->assertSame($logger, $this->fixture->logger());
    }

    #[Test]
    public function setLoggerCanClear() : void
    {
        $this->fixture->setLogger(new BlackHole());

        $this->fixture->setLogger(null);

        $this->assertNull($this->fixture->logger());
    }
}

/**
 * A fixture exposing the LoggerAware trait.
 *
 * @internal
 */
class LoggerAwareFixture
{
    use LoggerAware;
}
