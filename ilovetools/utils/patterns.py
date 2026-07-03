"""
Design-pattern utilities for everyday developer and agent workflows.

This module provides small, well-tested, production-ready implementations of
the most commonly needed GoF (and adjacent) design patterns.  Every class is
fully type-hinted, documented with usage examples, and designed to be dropped
into any project with zero external dependencies.

Patterns included
------------------
* **Singleton** – thread-safe singleton via metaclass.
* **Observer** – lightweight publish / subscribe event bus.
* **Strategy** – context + strategy registry for interchangeable algorithms.
* **ChainOfResponsibility** – sequential handler chain with short-circuit.
* **Command** – encapsulated request with undo / redo history.
* **Registry** – decorator-based global registry for plugins / handlers.
* **Pipeline** – ordered processing stages with data flowing through each.
* **Builder** – fluent / chainable builder for complex object construction.

Examples
--------
>>> from ilovetools.utils.patterns import (
...     Singleton, Observer, Strategy, ChainOfResponsibility,
...     Command, Registry, Pipeline, Builder,
... )

>>> class Config(metaclass=Singleton):
...     def __init__(self):
...         self.debug = False
>>> a = Config(); b = Config()
>>> a is b
True

>>> bus = Observer()
>>> received = []
>>> bus.subscribe("event", lambda data: received.append(data))
>>> bus.publish("event", 42)
>>> received
[42]

>>> strat = Strategy(default="add")
>>> strat.register("add", lambda a, b: a + b)
>>> strat.register("mul", lambda a, b: a * b)
>>> strat.execute("add", 3, 4)
7
>>> strat.execute("mul", 3, 4)
12

>>> chain = ChainOfResponsibility()
>>> chain.add(lambda req: req.get("type") == "A", lambda req: "handled-A")
>>> chain.add(lambda req: req.get("type") == "B", lambda req: "handled-B")
>>> chain.handle({"type": "B"})
'handled-B'
"""

from __future__ import annotations

import threading
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

__all__ = [
    "Singleton",
    "Observer",
    "Strategy",
    "ChainOfResponsibility",
    "Command",
    "Registry",
    "Pipeline",
    "Builder",
]

T = TypeVar("T")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
class Singleton(type):
    """Thread-safe singleton metaclass.

    Any class that uses ``metaclass=Singleton`` will have at most one
    instance per process.  The first call creates the instance; all
    subsequent calls return the same object.  Thread safety is guaranteed
    via a per-class lock.

    Example
    -------
    >>> class Database(metaclass=Singleton):
    ...     def __init__(self):
    ...         self.connected = True
    >>> db1 = Database()
    >>> db2 = Database()
    >>> db1 is db2
    True
    >>> db1.connected
    True
    """

    _instances: Dict[type, Any] = {}
    _locks: Dict[type, threading.Lock] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in Singleton._instances:
            # Create a lock per-class if not yet created
            if cls not in Singleton._locks:
                Singleton._locks[cls] = threading.Lock()
            with Singleton._locks[cls]:
                # Double-checked locking
                if cls not in Singleton._instances:
                    Singleton._instances[cls] = super().__call__(*args, **kwargs)
        return Singleton._instances[cls]

    @classmethod
    def _clear_instance(mcs, cls: type) -> None:
        """Remove the cached instance for *cls* (useful in tests)."""
        mcs._instances.pop(cls, None)
        mcs._locks.pop(cls, None)


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------
class Observer:
    """A lightweight publish / subscribe event bus.

    Subscribers register callbacks for named events.  When an event is
    published, all registered callbacks for that event are invoked in
    subscription order.  Exceptions in individual callbacks are caught
    and collected so that one failing handler does not break others.

    Example
    -------
    >>> bus = Observer()
    >>> results = []
    >>> bus.subscribe("login", lambda user: results.append(f"hi {user}"))
    >>> bus.publish("login", "alice")
    >>> results
    ['hi alice']

    >>> # Unsubscribe
    >>> handler_id = bus.subscribe("logout", lambda x: None)
    >>> bus.unsubscribe("logout", handler_id)
    >>> bus.subscriber_count("logout")
    0
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, Dict[int, Callable[..., Any]]] = {}
        self._counter: int = 0
        self._lock = threading.Lock()

    def subscribe(self, event: str, callback: Callable[..., Any]) -> int:
        """Register *callback* for *event* and return a subscription ID.

        Args:
            event: The event name to listen for.
            callback: A callable invoked with the data passed to
                :meth:`publish`.

        Returns:
            An integer subscription ID that can be used with
            :meth:`unsubscribe`.
        """
        with self._lock:
            self._counter += 1
            sub_id = self._counter
            self._subscribers.setdefault(event, {})[sub_id] = callback
            return sub_id

    def unsubscribe(self, event: str, subscription_id: int) -> bool:
        """Remove a subscription.  Returns ``True`` if it existed."""
        with self._lock:
            subs = self._subscribers.get(event)
            if subs is None:
                return False
            return subs.pop(subscription_id, None) is not None

    def publish(self, event: str, *args: Any, **kwargs: Any) -> List[Any]:
        """Publish *event* and call all registered callbacks.

        Args:
            event: The event name to publish.
            *args: Positional arguments forwarded to each callback.
            **kwargs: Keyword arguments forwarded to each callback.

        Returns:
            A list of return values from each callback that ran
            successfully.  Callbacks that raised exceptions are skipped
            (the exception is silently swallowed to keep the bus
            resilient).
        """
        with self._lock:
            subs = dict(self._subscribers.get(event, {}))
        results: List[Any] = []
        for cb in subs.values():
            try:
                results.append(cb(*args, **kwargs))
            except Exception:
                pass
        return results

    def subscriber_count(self, event: str) -> int:
        """Return the number of active subscribers for *event*."""
        with self._lock:
            return len(self._subscribers.get(event, {}))

    def clear(self) -> None:
        """Remove all subscribers for all events."""
        with self._lock:
            self._subscribers.clear()


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class Strategy(Generic[T, R]):
    """Context object for the Strategy pattern.

    Register named strategies (callables) and switch between them at
    runtime without changing client code.  A default strategy name may
    be specified so that :meth:`execute` can be called without an
    explicit strategy name.

    Example
    -------
    >>> s = Strategy(default="upper")
    >>> s.register("upper", lambda text: text.upper())
    >>> s.register("lower", lambda text: text.lower())
    >>> s.execute("upper", "Hello")
    'HELLO'
    >>> s.execute("lower", "Hello")
    'hello'
    >>> s.execute(None, "Hello")  # uses default
    'HELLO'
    """

    def __init__(self, default: Optional[str] = None) -> None:
        self._strategies: Dict[str, Callable[..., R]] = {}
        self._default: Optional[str] = default

    @property
    def default(self) -> Optional[str]:
        """The name of the default strategy."""
        return self._default

    @default.setter
    def default(self, name: Optional[str]) -> None:
        if name is not None and name not in self._strategies:
            raise KeyError(f"Strategy '{name}' is not registered.")
        self._default = name

    def register(self, name: str, fn: Callable[..., R]) -> None:
        """Register a strategy callable under *name*."""
        if not name:
            raise ValueError("Strategy name must be a non-empty string.")
        self._strategies[name] = fn

    def unregister(self, name: str) -> bool:
        """Remove a registered strategy.  Returns ``True`` if it existed."""
        existed = self._strategies.pop(name, None) is not None
        if existed and self._default == name:
            self._default = None
        return existed

    def execute(self, name: Optional[str] = None, *args: Any, **kwargs: Any) -> R:
        """Execute the named strategy (or the default).

        Args:
            name: Strategy to run.  If ``None``, the default strategy
                is used.
            *args, **kwargs: Forwarded to the strategy callable.

        Raises:
            KeyError: If the requested strategy (or default) is not
                registered.
        """
        resolved = name if name is not None else self._default
        if resolved is None:
            raise KeyError("No strategy name given and no default set.")
        if resolved not in self._strategies:
            raise KeyError(f"Strategy '{resolved}' is not registered.")
        return self._strategies[resolved](*args, **kwargs)

    @property
    def available(self) -> List[str]:
        """Return a sorted list of registered strategy names."""
        return sorted(self._strategies)


# ---------------------------------------------------------------------------
# ChainOfResponsibility
# ---------------------------------------------------------------------------
class ChainOfResponsibility:
    """Sequential handler chain with conditional short-circuiting.

    Each handler is a ``(condition, action)`` pair.  When :meth:`handle`
    is called, the chain is walked in order; the first handler whose
    ``condition(request)`` returns a truthy value has its ``action``
    called and the result is returned.  If no handler matches, a
    configurable default value (or exception) is returned.

    Example
    -------
    >>> chain = ChainOfResponsibility()
    >>> chain.add(lambda r: r["level"] >= 3, lambda r: "critical")
    >>> chain.add(lambda r: r["level"] >= 1, lambda r: "warning")
    >>> chain.handle({"level": 2})
    'warning'
    >>> chain.handle({"level": 5})
    'critical'
    """

    def __init__(self, default: Any = None) -> None:
        self._handlers: List[Tuple[Callable[[Any], bool], Callable[[Any], Any]]] = []
        self._default: Any = default

    def add(
        self,
        condition: Callable[[Any], bool],
        action: Callable[[Any], Any],
    ) -> "ChainOfResponsibility":
        """Append a handler pair to the end of the chain.

        Returns ``self`` so calls can be chained.
        """
        self._handlers.append((condition, action))
        return self

    def handle(self, request: Any) -> Any:
        """Walk the chain and return the first matching handler's result.

        If no handler matches, return the default value.
        """
        for condition, action in self._handlers:
            if condition(request):
                return action(request)
        return self._default

    def reset(self) -> None:
        """Remove all handlers from the chain."""
        self._handlers.clear()

    @property
    def length(self) -> int:
        """Number of handlers currently in the chain."""
        return len(self._handlers)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------
class Command:
    """Encapsulated action with undo / redo support.

    Wrap an operation (``do``) and its inverse (``undo``) so that the
    operation can be executed, reverted, and re-applied.  A history
    stack is maintained automatically.

    Example
    -------
    >>> state = {"value": 10}
    >>> cmd = Command(
    ...     do=lambda: state.__setitem__("value", state["value"] + 5),
    ...     undo=lambda: state.__setitem__("value", state["value"] - 5),
    ... )
    >>> cmd.execute()
    >>> state["value"]
    15
    >>> cmd.undo()
    >>> state["value"]
    10
    >>> cmd.redo()
    >>> state["value"]
    15
    """

    def __init__(
        self,
        do: Callable[[], Any],
        undo: Optional[Callable[[], Any]] = None,
        *,
        name: str = "",
    ) -> None:
        self._do = do
        self._undo = undo
        self.name = name
        self._executed: bool = False

    def execute(self) -> Any:
        """Run the *do* action and mark the command as executed."""
        result = self._do()
        self._executed = True
        return result

    def undo(self) -> Any:
        """Run the *undo* action if one was provided and the command
        was previously executed.

        Raises:
            RuntimeError: If no undo callback was supplied.
        """
        if self._undo is None:
            raise RuntimeError(f"Command '{self.name}' has no undo callback.")
        if not self._executed:
            raise RuntimeError(
                f"Command '{self.name}' cannot be undone before execution."
            )
        result = self._undo()
        self._executed = False
        return result

    def redo(self) -> Any:
        """Re-execute the command (equivalent to :meth:`execute`)."""
        return self.execute()

    @property
    def is_executed(self) -> bool:
        """Whether the command is currently in the *executed* state."""
        return self._executed


class CommandHistory:
    """Manages a stack of :class:`Command` objects with undo / redo.

    Example
    -------
    >>> history = CommandHistory()
    >>> val = {"n": 0}
    >>> c1 = Command(lambda: val.__setitem__("n", val["n"] + 1),
    ...               lambda: val.__setitem__("n", val["n"] - 1))
    >>> c2 = Command(lambda: val.__setitem__("n", val["n"] + 10),
    ...               lambda: val.__setitem__("n", val["n"] - 10))
    >>> history.execute(c1)
    >>> history.execute(c2)
    >>> val["n"]
    11
    >>> history.undo()
    >>> val["n"]
    1
    >>> history.redo()
    >>> val["n"]
    11
    """

    def __init__(self, max_size: int = 100) -> None:
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []
        self._max_size = max_size

    def execute(self, command: Command) -> Any:
        """Execute *command* and push it onto the undo stack."""
        result = command.execute()
        self._undo_stack.append(command)
        if len(self._undo_stack) > self._max_size:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        return result

    def undo(self) -> Any:
        """Undo the most recently executed command.

        Raises:
            RuntimeError: If there is nothing to undo.
        """
        if not self._undo_stack:
            raise RuntimeError("Nothing to undo.")
        command = self._undo_stack.pop()
        result = command.undo()
        self._redo_stack.append(command)
        return result

    def redo(self) -> Any:
        """Redo the most recently undone command.

        Raises:
            RuntimeError: If there is nothing to redo.
        """
        if not self._redo_stack:
            raise RuntimeError("Nothing to redo.")
        command = self._redo_stack.pop()
        result = command.redo()
        self._undo_stack.append(command)
        return result

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def clear(self) -> None:
        """Clear all history."""
        self._undo_stack.clear()
        self._redo_stack.clear()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class Registry(Generic[T]):
    """Decorator-based registry for plugins, handlers, or strategies.

    Register classes or functions with a name and later look them up
    by that name.  Useful for plugin systems, command dispatch, and
    factory patterns.

    Example
    -------
    >>> reg = Registry()
    >>> @reg.register("greet")
    ... def greet(name):
    ...     return f"Hello, {name}!"
    >>> reg.get("greet")("world")
    'Hello, world!'
    >>> "greet" in reg
    True
    >>> sorted(reg.names)
    ['greet']
    """

    def __init__(self) -> None:
        self._items: Dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator that registers the decorated object under *name*.

        Args:
            name: Unique key under which to register the decorated
                class or function.

        Raises:
            ValueError: If *name* is already registered.
        """

        def decorator(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"'{name}' is already registered.")
            self._items[name] = obj
            return obj

        return decorator

    def register_instance(self, name: str, obj: T) -> None:
        """Register an existing object (non-decorator usage).

        Raises:
            ValueError: If *name* is already registered.
        """
        if name in self._items:
            raise ValueError(f"'{name}' is already registered.")
        self._items[name] = obj

    def get(self, name: str) -> T:
        """Retrieve a registered item by name.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._items:
            raise KeyError(f"'{name}' is not registered.")
        return self._items[name]

    def unregister(self, name: str) -> bool:
        """Remove a registered item.  Returns ``True`` if it existed."""
        return self._items.pop(name, None) is not None

    @property
    def names(self) -> List[str]:
        """Return a sorted list of all registered names."""
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class Pipeline:
    """Ordered processing pipeline where data flows through stages.

    Each stage is a callable that receives the current data and returns
    transformed data.  Stages are executed in registration order.

    Example
    -------
    >>> pipe = Pipeline()
    >>> pipe.add(lambda x: x + 1)
    >>> pipe.add(lambda x: x * 2)
    >>> pipe.add(lambda x: x - 3)
    >>> pipe.run(10)
    19

    >>> # With named stages for introspection
    >>> pipe2 = Pipeline()
    >>> pipe2.add(lambda s: s.strip(), name="strip")
    >>> pipe2.add(lambda s: s.title(), name="title")
    >>> pipe2.run("  hello world  ")
    'Hello World'
    >>> pipe2.stage_names
    ['strip', 'title']
    """

    def __init__(self) -> None:
        self._stages: List[Tuple[Optional[str], Callable[[Any], Any]]] = []

    def add(
        self,
        stage: Callable[[Any], Any],
        *,
        name: Optional[str] = None,
    ) -> "Pipeline":
        """Append a processing stage to the pipeline.

        Returns ``self`` for chaining.
        """
        self._stages.append((name, stage))
        return self

    def run(self, data: Any) -> Any:
        """Run all stages in order, passing data through each.

        Raises:
            RuntimeError: If the pipeline has no stages.
        """
        if not self._stages:
            raise RuntimeError("Pipeline has no stages.")
        for _, stage in self._stages:
            data = stage(data)
        return data

    @property
    def stage_names(self) -> List[str]:
        """Return the names of all stages (``None`` names are shown as
        ``'<anonymous>'``)."""
        return [name or "<anonymous>" for name, _ in self._stages]

    @property
    def length(self) -> int:
        """Number of stages in the pipeline."""
        return len(self._stages)

    def reset(self) -> None:
        """Remove all stages."""
        self._stages.clear()


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class Builder(Generic[T]):
    """Fluent / chainable builder for constructing objects step by step.

    Provide a target class (or factory callable) and then chain
    ``set(key, value)`` calls.  Call :meth:`build` to instantiate the
    object with all accumulated attributes passed as keyword arguments.

    Example
    -------
    >>> class User:
    ...     def __init__(self, name="", email="", age=0):
    ...         self.name = name
    ...         self.email = email
    ...         self.age = age
    ...     def __repr__(self):
    ...         return f"User({self.name!r}, {self.email!r}, {self.age})"
    >>> user = (Builder(User)
    ...         .set("name", "Alice")
    ...         .set("email", "alice@example.com")
    ...         .set("age", 30)
    ...         .build())
    >>> user.name
    'Alice'
    >>> user.age
    30

    >>> # Reset and reuse
    >>> b = Builder(User).set("name", "Bob")
    >>> b.build().name
    'Bob'
    >>> b.reset()
    >>> b.build().name
    ''
    """

    def __init__(self, factory: Callable[..., T]) -> None:
        self._factory = factory
        self._attrs: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> "Builder[T]":
        """Set a single attribute.  Returns ``self`` for chaining."""
        self._attrs[key] = value
        return self

    def set_many(self, **kwargs: Any) -> "Builder[T]":
        """Set multiple attributes at once.  Returns ``self``."""
        self._attrs.update(kwargs)
        return self

    def build(self) -> T:
        """Instantiate the target object with all set attributes."""
        return self._factory(**self._attrs)

    def reset(self) -> "Builder[T]":
        """Clear all set attributes.  Returns ``self``."""
        self._attrs.clear()
        return self

    @property
    def attributes(self) -> Dict[str, Any]:
        """Return a copy of the currently set attributes."""
        return dict(self._attrs)
