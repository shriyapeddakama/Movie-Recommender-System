from unittest.mock import MagicMock, patch
from shrekommender_system.services.recommender_service import RecommenderService

def test_recommend_returns_model_result():
    mock_router = MagicMock()
    mock_router.select.return_value = "als@v1"
    mock_router.fallback.return_value = None

    mock_model = MagicMock()
    mock_model.recommend_for_user.return_value = ["movie1", "movie2"]

    mock_registry = MagicMock()
    mock_registry.ensure_loaded.return_value = mock_model
    mock_registry.list_loaded.return_value = {"als@v1": mock_model}

    with patch("shrekommender_system.services.recommender_service.ModelRouter", return_value=mock_router), \
         patch("shrekommender_system.services.recommender_service.ModelRegistry", return_value=mock_registry):
        
        service = RecommenderService()
        result = service.recommend("user123", top_k=2)

        assert result == ["movie1", "movie2"]
        mock_router.select.assert_called_once()
        mock_registry.ensure_loaded.assert_called_with("als@v1")
        mock_model.recommend_for_user.assert_called_with("user123", 2)

def test_recommend_fallback_if_primary_fails():
    mock_router = MagicMock()
    mock_router.select.return_value = "primary_model"
    mock_router.fallback.return_value = "fallback_model"

    primary_model = MagicMock()
    primary_model.recommend_for_user.side_effect = Exception("fail")
    fallback_model = MagicMock()
    fallback_model.recommend_for_user.return_value = ["fallback_movie"]

    mock_registry = MagicMock()
    mock_registry.ensure_loaded.side_effect = lambda model_id: primary_model if model_id == "primary_model" else fallback_model
    mock_registry.list_loaded.return_value = {"primary_model": primary_model, "fallback_model": fallback_model}

    with patch("shrekommender_system.services.recommender_service.ModelRouter", return_value=mock_router), \
         patch("shrekommender_system.services.recommender_service.ModelRegistry", return_value=mock_registry):
        service = RecommenderService()
        result = service.recommend("user123", top_k=1)
        assert result == ["fallback_movie"]

def test_health_returns_loaded_models_and_router(monkeypatch):
    mock_router = MagicMock()
    mock_router.config.default_model = "als@v1"
    mock_router.config.rules = []

    mock_model = MagicMock()
    mock_model.health_check.return_value = {"status": "healthy"}

    mock_registry = MagicMock()
    mock_registry.list_loaded.return_value = {"als@v1": mock_model}

    monkeypatch.setattr("shrekommender_system.services.recommender_service.ModelRouter", lambda *a, **kw: mock_router)
    monkeypatch.setattr("shrekommender_system.services.recommender_service.ModelRegistry", lambda *a, **kw: mock_registry)

    service = RecommenderService()
    health_stats = service.health()
    assert health_stats["models"]["als@v1"] == {"status": "healthy"}
    assert health_stats["router"]["default_model"] == "als@v1"
